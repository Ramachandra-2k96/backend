import io
import os
import base64
import csv
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torchvision.transforms as T

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import timm

app = FastAPI(title="Pathology Multi-Task Classification API")

# Default classes from manifest structure
DEFAULT_CANCER_CLASSES = [
    "brain_glioblastoma", "brain_lower_grade_glioma", "breast_cancer",
    "colon_adenocarcinoma", "lung_adenocarcinoma", "lung_squamous_cell",
    "prostate_cancer", "rectal_adenocarcinoma"
]
DEFAULT_SUBTYPE_CLASSES = [
    "Metastatic", "Primary Tumor", "Recurrent Tumor", "Solid Tissue Normal"
]

def load_classes_from_manifest(manifest_path: str = "../data/tcga_train_data/manifest.csv") -> Tuple[List[str], List[str]]:
    """Attempt to load unique classes dynamically from manifest."""
    if not os.path.exists(manifest_path):
        return sorted(DEFAULT_CANCER_CLASSES), sorted(DEFAULT_SUBTYPE_CLASSES)
    
    cancers, subtypes = set(), set()
    try:
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                c = row.get("cancer_type", "").strip()
                s = row.get("sample_type", "").strip()
                if c: cancers.add(c)
                if s: subtypes.add(s)
        if cancers and subtypes:
            return sorted(list(cancers)), sorted(list(subtypes))
    except Exception as e:
        print(f"Error reading manifest: {e}")
    
    return sorted(DEFAULT_CANCER_CLASSES), sorted(DEFAULT_SUBTYPE_CLASSES)


# ──────────────────────────────────────────────────────────────────
# Model Architecture (Identical to Train.py)
# ──────────────────────────────────────────────────────────────────
class DualHeadModel(nn.Module):
    def __init__(self, backbone_name: str, num_cancer: int, num_subtype: int):
        super().__init__()
        if backbone_name == "vit_small":
            self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
            feat_dim = self.backbone.num_features
        elif backbone_name == "resnet50":
            self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0, global_pool="avg")
            feat_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unknown MODEL_NAME '{backbone_name}'")

        self.head_cancer = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, num_cancer))
        self.head_subtype = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, num_subtype))

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_cancer(feat), self.head_subtype(feat)


# ──────────────────────────────────────────────────────────────────
# Wrapper for GradCAM (Targets the cancer head)
# ──────────────────────────────────────────────────────────────────
class GradCamModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        cancer_logits, _ = self.model(x)
        return cancer_logits


# ──────────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_model_name = None
model_instance = None
cancer_classes = []
subtype_classes = []

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

inference_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def get_target_layer(model, model_name):
    """Get the appropriate target layer for GradCAM based on the architecture."""
    if model_name == "vit_small":
        return [model.backbone.blocks[-1].norm1]
    elif model_name == "resnet50":
        return [model.backbone.layer4[-1]]
    return None

def load_model(model_name: str):
    """Loads model and weights dynamically."""
    global current_model_name, model_instance, cancer_classes, subtype_classes

    if current_model_name == model_name and model_instance is not None:
        return model_instance

    cancer_classes, subtype_classes = load_classes_from_manifest()
    
    model = DualHeadModel(model_name, len(cancer_classes), len(subtype_classes))
    
    # Try to load latest or best weights if they exist
    ckpt_paths = [
        f"../data/checkpoints/{model_name}_best.pth",
        f"../data/checkpoints/{model_name}_latest.pth"
    ]
    
    loaded = False
    for ckpt_path in ckpt_paths:
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                # Handle both EMA and normal weights
                if "ema_state_dict" in ckpt:
                    model.load_state_dict(ckpt["ema_state_dict"])
                elif "state_dict" in ckpt:
                    model.load_state_dict(ckpt["state_dict"])
                else:
                    model.load_state_dict(ckpt)
                print(f"Loaded weights from {ckpt_path}")
                loaded = True
                break
            except Exception as e:
                print(f"Failed to load {ckpt_path}: {e}")
                
    if not loaded:
        print(f"No saved weights found for {model_name}. Using untrained pretrained backbone.")

    model = model.to(device)
    model.eval()
    
    current_model_name = model_name
    model_instance = model
    return model_instance

def reshape_transform(tensor, height=14, width=14):
    # Tensor shape is [batch, seq_len, dim]. For 224x224 patch16, seq_len is 197 (1 cls token + 196 patches)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def generate_gradcam(model, model_name, input_tensor, original_img_np, cancer_pred_idx):
    target_layers = get_target_layer(model, model_name)
    if not target_layers:
        return None
        
    cam_model = GradCamModelWrapper(model)
    
    # Use reshape_transform for ViT models
    if "vit" in model_name:
        cam = GradCAM(model=cam_model, target_layers=target_layers, reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=cam_model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(cancer_pred_idx)]
    
    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert original image to 0-1 range float for show_cam_on_image
        rgb_img = original_img_np.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return cam_image
    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("vit_small")
):
    if model_name not in ["vit_small", "resnet50"]:
        raise HTTPException(status_code=400, detail="Invalid model_name. Choose 'vit_small' or 'resnet50'")

    # Read and process image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # For GradCAM visualization, we need the original image resized to 224x224
    img_resized = image.resize((224, 224))
    img_np = np.array(img_resized)

    # Transform for inference
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Load model
    model = load_model(model_name)

    # Inference
    with torch.no_grad():
        cancer_logits, subtype_logits = model(input_tensor)
        
        cancer_probs = torch.softmax(cancer_logits, dim=1)[0]
        subtype_probs = torch.softmax(subtype_logits, dim=1)[0]
        
        cancer_pred_idx = torch.argmax(cancer_probs).item()
        subtype_pred_idx = torch.argmax(subtype_probs).item()

    predicted_cancer = cancer_classes[cancer_pred_idx]
    predicted_subtype = subtype_classes[subtype_pred_idx]
    
    # Generate GradCAM
    # We must enable gradients for GradCAM
    input_tensor.requires_grad_(True)
    cam_img_np = generate_gradcam(model, model_name, input_tensor, img_np, cancer_pred_idx)
    
    response_data = {
        "cancer_type": {
            "prediction": predicted_cancer,
            "confidence": float(cancer_probs[cancer_pred_idx].item())
        },
        "sample_type": {
            "prediction": predicted_subtype,
            "confidence": float(subtype_probs[subtype_pred_idx].item())
        },
        "gradcam_base64": None
    }
    
    # Encode GradCAM image to base64
    if cam_img_np is not None:
        cam_pil = Image.fromarray(cam_img_np)
        buffered = io.BytesIO()
        cam_pil.save(buffered, format="JPEG")
        cam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response_data["gradcam_base64"] = cam_b64

    return JSONResponse(content=response_data)
