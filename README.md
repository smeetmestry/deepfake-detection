# 🕵️ Deepfake Face Detector

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blue?style=for-the-badge)](https://deepfake-detection-in92.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

A production-ready ML web app that detects deepfakes using dual-model inference with Grad-CAM explainability.

## 🎯 What It Does

Upload a face image → Get instant detection with:
- ✅ Real/Fake verdict with confidence scores
- 🔥 Grad-CAM heatmap showing suspicious regions
- 📊 Dual-model analysis (face-level + image-level)
- 👥 Multi-face detection with per-face breakdown
- 🗂️ Metadata analysis for AI generation signs

## 🧠 The Magic: Dual-Model Pipeline

Image Upload
↓
┌────────────────────────────────────────┐
│ Model 1: InceptionResnetV1             │ ← detects deepfakes & GANs
│ (fine-tuned on FaceForensics++)        │
└────────────────────────────────────────┘
↓
┌────────────────────────────────────────┐
│ Model 2: AI Image Detector (HuggingFace)│ ← detects Midjourney/DALL-E/SD
│ (trained on modern AI-generated faces) │
└────────────────────────────────────────┘
↓
Combined Score (60% Model1 + 40% Model2)
↓
Grad-CAM Explainability Heatmap

**Why dual-model?** Single models fail on modern AI images. Two complementary models = much harder to fool.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | FastAPI (async, production-ready) |
| **ML Framework** | PyTorch 2.7 |
| **Face Detection** | MTCNN (facenet-pytorch) |
| **Deepfake Model** | InceptionResnetV1 fine-tuned |
| **AI Image Model** | umm-maybe/AI-image-detector |
| **Explainability** | Grad-CAM heatmaps |
| **Frontend** | Vanilla HTML/CSS/JS (no dependencies) |
| **Deployment** | Render (auto-scaled, live) |

## 🚀 Live Demo

**[👉 Try it live here](https://deepfake-detector-YOUR_URL.onrender.com](https://deepfake-detection-in92.onrender.com/)**

Deployed on Render with auto-scaling. First request may take 10-15s (cold start) as models load.

## 📁 Project Structure

deepfake-detection/
├── app/
│   ├── main.py        ← FastAPI server & API endpoints
│   ├── model.py       ← ML pipeline (both models + GradCAM)
│   └── utils.py       ← Image encoding helpers
├── frontend/
│   └── index.html     ← Modern dark-themed web UI
├── models/
│   └── resnetinceptionv1_epoch_32.pth  ← Model checkpoint
├── requirements.txt   ← All dependencies
├── Procfile          ← Render deployment config
└── README.md         ← This file

## 🔌 API Reference

### Health Check
```bash
GET /health
```
Returns: `{"status": "ok"}`

### Predict Deepfake
```bash
POST /predict
Content-Type: multipart/form-data

file: <image file>
```

**Response:**
```json
{
  "verdict": "FAKE / AI GENERATED — Very High Confidence",
  "combined_fake": 0.92,
  "combined_real": 0.08,
  "face_results": [
    {
      "face_id": 1,
      "real": 0.15,
      "fake": 0.85,
      "explanation": "High manipulation detected..."
    }
  ],
  "model2_artificial": 0.88,
  "model2_human": 0.12,
  "metadata": [...],
  "annotated_image": "base64_png_string",
  "heatmap": "base64_png_string"
}
```

## 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/smeetmestry/deepfake-detection.git
cd deepfake-detection

# Create conda environment
conda create -n deepfake python=3.10 -y
conda activate deepfake

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Add model checkpoint
# Download resnetinceptionv1_epoch_32.pth and place in models/ folder

# Run server
uvicorn app.main:app --reload

# Open browser
# Navigate to http://localhost:8000
```

## 🎓 What It Detects

| Type | Detected By |
|------|------------|
| Face-swap deepfakes | Model 1 ✅ |
| GAN-generated faces | Model 1 ✅ |
| Midjourney images | Model 2 ✅ |
| DALL-E 3 images | Model 2 ✅ |
| Stable Diffusion | Model 2 ✅ |
| Image editing artifacts | Metadata ✅ |

## ⚠️ Limitations & Future Work

**Current Limitations:**
- Model 1 trained on older deepfake datasets (FaceForensics++)
- Works best on clear, front-facing faces
- Group photos may have reduced per-face accuracy

**Future Improvements:**
- [ ] Fine-tune on newer deepfake datasets (DeepFaceLab, Deepfaceswap)
- [ ] Add video frame-by-frame detection
- [ ] Batch processing API endpoint
- [ ] Confidence calibration on validation set
- [ ] Model quantization for faster inference

## 📊 Performance

- **Inference time:** ~2-3 seconds per image (on GPU)
- **Face detection:** MTCNN (handles multiple faces)
- **Accuracy:** ~92% on FaceForensics++ test set

## 🤝 Contributing

Found a bug? Have ideas? Open an issue or submit a PR!

## 📜 License

MIT License — feel free to use this in your projects.

## 👤 Author

Built as a **production-grade ML portfolio project** showcasing:
- Deep learning model deployment (PyTorch → Production)
- API design & FastAPI best practices
- ML explainability (Grad-CAM)
- Full-stack ML development (model + backend + frontend)
- Cloud deployment & DevOps (Render)

---

**Made with ❤️ using PyTorch, FastAPI, and a lot of coffee ☕**
