from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
from app.model import predict
from app.utils import image_to_base64

app = FastAPI(title="Deepfake Detector API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def run_prediction(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg","image/png","image/webp"]:
        raise HTTPException(400, f"Invalid file type: {file.content_type}")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")
    try:
        r = predict(image)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    return JSONResponse({
        "verdict": r["verdict"],
        "combined_fake": r["combined_fake"],
        "combined_real": r["combined_real"],
        "face_results": r["face_results"],
        "model2_artificial": r["model2_artificial"],
        "model2_human": r["model2_human"],
        "metadata": r["metadata"],
        "annotated_image": image_to_base64(r["annotated_image"]),
        "heatmap": image_to_base64(r["heatmap"])
    })