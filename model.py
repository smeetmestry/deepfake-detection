import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import pipeline
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

mtcnn_multi = MTCNN(keep_all=True, post_process=False, device=DEVICE).to(DEVICE).eval()

model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "models", "resnetinceptionv1_epoch_32.pth")
if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print("Model 1 loaded!")
else:
    print(f"ERROR: checkpoint not found at {CHECKPOINT}")

print("Loading Model 2...")
ai_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector",
                        device=0 if torch.cuda.is_available() else -1)
print("Model 2 loaded!")

def analyze_single_face(face_tensor):
    face = face_tensor.unsqueeze(0)
    face = F.interpolate(face, size=(256,256), mode="bilinear", align_corners=False)
    prev_face = face.squeeze(0).permute(1,2,0).cpu().detach().numpy().astype("uint8")
    face = face.to(DEVICE).to(torch.float32) / 255.0
    face_plot = np.clip(face.squeeze(0).permute(1,2,0).cpu().detach().numpy(), 0, 1)
    cam = GradCAM(model=model, target_layers=[model.block8.branch1[-1]])
    gcam = cam(input_tensor=face, targets=[ClassifierOutputTarget(0)], eigen_smooth=True)[0]
    viz = show_cam_on_image(face_plot, gcam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, viz, 0.5, 0)
    with torch.no_grad():
        out = torch.sigmoid(model(face).squeeze(0)).item()
    return round(1-out,4), round(out,4), face_with_mask

def get_explanation(s):
    if s >= 0.85: return "High manipulation detected. Suspicious regions in skin, eyes, or boundaries."
    elif s >= 0.60: return "Moderate anomalies. Inconsistencies typical of AI generation."
    elif s >= 0.40: return "Borderline. Minor inconsistencies, could be real."
    else: return "No significant manipulation. Looks natural."

def check_metadata(image):
    info = image.info
    w, h = image.size
    ai_sizes = [(512,512),(1024,1024),(768,768),(1024,768),(832,1216)]
    return [
        "No metadata — common in AI images" if not info else f"Metadata present ({len(info)} fields)",
        "Transparency channel — possible editing" if image.mode=="RGBA" else f"Image mode: {image.mode}",
        f"Resolution {w}x{h} matches AI generation sizes" if (w,h) in ai_sizes else f"Resolution {w}x{h} — not standard AI size"
    ]

def predict(image: Image.Image):
    ai_result = ai_detector(image)
    ai_scores = {r["label"]: r["score"] for r in ai_result}
    ai_fake = ai_scores.get("artificial", 0)
    ai_real = ai_scores.get("human", 0)

    boxes, _ = mtcnn_multi.detect(image)
    faces = mtcnn_multi(image)
    face_results = []
    annotated = image.copy()
    final_heatmap = image.copy()

    if faces is not None and boxes is not None:
        draw = ImageDraw.Draw(annotated)
        all_heatmaps = []
        for i, (ft, box) in enumerate(zip(faces, boxes)):
            rp, fp, hm = analyze_single_face(ft)
            face_results.append({"face_id":i+1,"real":rp,"fake":fp,"explanation":get_explanation(fp)})
            all_heatmaps.append(Image.fromarray(hm))
            color = "red" if fp>=0.60 else ("orange" if fp>=0.40 else "green")
            x1,y1,x2,y2 = [int(b) for b in box]
            draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
            draw.rectangle([x1,y1-22,x2,y1], fill=color)
            draw.text((x1+4,y1-20), f"Face {i+1}: {fp*100:.0f}% fake", fill="white")
        avg_fake = sum(f["fake"] for f in face_results)/len(face_results)
        combined_fake = (0.6*avg_fake)+(0.4*ai_fake)
        if len(all_heatmaps)==1:
            final_heatmap = all_heatmaps[0]
        else:
            tw = sum(h.width for h in all_heatmaps)
            mh = max(h.height for h in all_heatmaps)
            final_heatmap = Image.new("RGB",(tw,mh))
            xo=0
            for hm in all_heatmaps:
                final_heatmap.paste(hm,(xo,0)); xo+=hm.width
    else:
        combined_fake = ai_fake

    combined_real = 1 - combined_fake
    if combined_fake>=0.85: verdict="FAKE / AI GENERATED — Very High Confidence"
    elif combined_fake>=0.65: verdict="LIKELY FAKE / AI GENERATED"
    elif combined_fake>=0.45: verdict="UNCERTAIN"
    elif combined_real>=0.80: verdict="LIKELY REAL"
    else: verdict="REAL — High Confidence"

    return {
        "verdict": verdict,
        "combined_fake": round(combined_fake,4),
        "combined_real": round(combined_real,4),
        "face_results": face_results,
        "model2_artificial": round(ai_fake,4),
        "model2_human": round(ai_real,4),
        "metadata": check_metadata(image),
        "annotated_image": annotated,
        "heatmap": final_heatmap
    }