import io
import base64
import csv
from pathlib import Path
from typing import Tuple, List
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torchvision.transforms as T

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import timm

app = FastAPI(title="Pathology Multi-Task Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MANIFEST_PATH = Path(__file__).parent / "manifest.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

inference_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_classes_from_manifest() -> Tuple[List[str], List[str]]:
    cancers, subtypes = set(), set()
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            c = row.get("cancer_type", "").strip()
            s = row.get("sample_type", "").strip()
            if c: cancers.add(c)
            if s: subtypes.add(s)
    return sorted(cancers), sorted(subtypes)


class DualHeadModel(nn.Module):
    def __init__(self, backbone_name: str, num_cancer: int, num_subtype: int):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "vit_small":
            self.backbone = timm.create_model(
                "vit_small_patch16_224", pretrained=False, num_classes=0, drop_path_rate=0.1
            )
            feat_dim = self.backbone.num_features
        elif backbone_name == "resnet50":
            self.backbone = timm.create_model(
                "resnet50", pretrained=False, num_classes=0, global_pool=""
            )
            feat_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unknown backbone '{backbone_name}'")

        self.head_cancer = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(0.3), nn.Linear(feat_dim, num_cancer))

        self.query_proj = nn.Linear(num_cancer, feat_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8, batch_first=True)

        self.head_subtype = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(0.3), nn.Linear(feat_dim, num_subtype))

    def forward(self, x):
        raw_feat = self.backbone.forward_features(x)

        if self.backbone_name == "vit_small":
            cls_feat     = raw_feat[:, 0]
            spatial_feat = raw_feat[:, 1:]
        else:
            cls_feat     = raw_feat.mean(dim=[2, 3])
            spatial_feat = raw_feat.flatten(2).transpose(1, 2)

        cancer = self.head_cancer(cls_feat)

        query         = self.query_proj(cancer.detach()).unsqueeze(1)
        attn_output, _ = self.cross_attn(query, spatial_feat, spatial_feat)
        subtype = self.head_subtype(attn_output.squeeze(1))

        return cancer, subtype


class GradCamModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        cancer_logits, _ = self.model(x)
        return cancer_logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_model_name = None
model_instance     = None
cancer_classes: List[str] = []
subtype_classes: List[str] = []


def get_target_layer(model, model_name):
    if model_name == "vit_small":
        return [model.backbone.blocks[-1].norm1]
    elif model_name == "resnet50":
        return [model.backbone.layer4[-1]]
    return None


def load_model(model_name: str):
    global current_model_name, model_instance, cancer_classes, subtype_classes

    if current_model_name == model_name and model_instance is not None:
        return model_instance

    cancer_classes, subtype_classes = load_classes_from_manifest()
    model = DualHeadModel(model_name, len(cancer_classes), len(subtype_classes))

    here = Path(__file__).parent
    for ckpt_name in [f"{model_name}_best.pth", f"{model_name}_latest.pth"]:
        ckpt_path = here / ckpt_name
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "ema_state_dict" in ckpt:
                state = ckpt["ema_state_dict"]
            else:
                state = ckpt
            model.load_state_dict(state)
            print(f"Loaded weights from {ckpt_path}")
            break

    model = model.to(device).eval()
    current_model_name = model_name
    model_instance = model
    return model_instance


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def generate_gradcam(model, model_name, input_tensor, original_img_np, cancer_pred_idx):
    target_layers = get_target_layer(model, model_name)
    if not target_layers:
        return None

    cam_model = GradCamModelWrapper(model)
    kwargs = {"reshape_transform": reshape_transform} if "vit" in model_name else {}
    cam = GradCAM(model=cam_model, target_layers=target_layers, **kwargs)

    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(cancer_pred_idx)])[0]
        return show_cam_on_image(original_img_np.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"GradCAM failed: {e}")
        return None


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("vit_small"),
):
    if model_name not in ["vit_small", "resnet50"]:
        raise HTTPException(status_code=400, detail="Invalid model_name. Choose 'vit_small' or 'resnet50'")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    img_np       = np.array(image.resize((224, 224)))
    input_tensor = inference_transform(image).unsqueeze(0).to(device)
    model        = load_model(model_name)

    with torch.no_grad():
        cancer_logits, subtype_logits = model(input_tensor)
        cancer_probs  = torch.softmax(cancer_logits, dim=1)[0]
        subtype_probs = torch.softmax(subtype_logits, dim=1)[0]
        cancer_pred_idx  = int(torch.argmax(cancer_probs).item())
        subtype_pred_idx = int(torch.argmax(subtype_probs).item())

    # GradCAM needs its own forward pass with gradients enabled.
    # The tensor used for inference went through no_grad; create a fresh one.
    cam_tensor = input_tensor.detach().clone().requires_grad_(True)
    cam_img_np = generate_gradcam(model, model_name, cam_tensor, img_np, cancer_pred_idx)

    gradcam_b64 = None
    if cam_img_np is not None:
        buf = io.BytesIO()
        Image.fromarray(cam_img_np).save(buf, format="JPEG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "cancer_type":  {"prediction": cancer_classes[cancer_pred_idx],  "confidence": float(cancer_probs[cancer_pred_idx])},
        "sample_type":  {"prediction": subtype_classes[subtype_pred_idx], "confidence": float(subtype_probs[subtype_pred_idx])},
        "gradcam_base64": gradcam_b64,
    })
