# Pathology Multi-Task Classification API

This backend provides a FastAPI-based service for classifying pathology images into cancer types and sample types, utilizing models like ViT or ResNet50. It also returns a Grad-CAM visualization for interpretability.

## API Endpoint Reference

### `POST /predict`

**Input (Form-Data)**
- `file`: The pathology image file (e.g., .jpg, .png) to be analyzed.
- `model_name` (optional): The architecture to use. Accepted values are `"vit_small"` or `"resnet50"`. Defaults to `"vit_small"`.

**Output (JSON)**
```json
{
  "cancer_type": {
    "prediction": "lung_adenocarcinoma",
    "confidence": 0.985
  },
  "sample_type": {
    "prediction": "Primary Tumor",
    "confidence": 0.991
  },
  "gradcam_base64": "/9j/4AAQSkZJRgABAQ..." // Base64 encoded JPEG of the Grad-CAM visualization
}
```

## How to use in Frontend

You can call this API from your frontend using standard multipart form-data. Here is an example using JavaScript's `fetch` API:

```javascript
// 1. Prepare the form data
const formData = new FormData();
const fileInput = document.getElementById("image-upload-input");

formData.append("file", fileInput.files[0]);
formData.append("model_name", "vit_small"); // Optional: choose "vit_small" or "resnet50"

// 2. Make the request
fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => {
    // 3. Handle the predictions
    console.log("Cancer Type:", data.cancer_type.prediction);
    console.log("Confidence:", data.cancer_type.confidence.toFixed(2));
    
    console.log("Sample Type:", data.sample_type.prediction);
    console.log("Confidence:", data.sample_type.confidence.toFixed(2));

    // 4. Display the Grad-CAM visualization
    if (data.gradcam_base64) {
      const imgElement = document.getElementById("gradcam-result");
      imgElement.src = `data:image/jpeg;base64,${data.gradcam_base64}`;
    }
  })
  .catch((error) => {
    console.error("Error analyzing image:", error);
  });
```
