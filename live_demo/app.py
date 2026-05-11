"""
Sikap-Salita live demo FastAPI server.

Start with:
    cd scratch/PANSINAYAN
    .venv/Scripts/python -m uvicorn live_demo.app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from live_demo.keypoints import extract_from_base64, get_models, shutdown_models
from live_demo.inference import push_frame, predict_top3, buffered_frame_count
from live_demo.model import load_model
from live_demo.labels import load_labels

_STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load model and MediaPipe so the first request isn't slow
    load_model()
    get_models()
    load_labels()
    print("[Sikap-Salita] Ready — open http://localhost:8000")
    yield
    shutdown_models()


app = FastAPI(title="Sikap-Salita Live Demo", lifespan=lifespan)


class PredictRequest(BaseModel):
    frame: str  # base64 JPEG, with or without data-URI prefix


@app.post("/predict")
async def predict(req: PredictRequest) -> JSONResponse:
    vec178, mask89 = extract_from_base64(req.frame)
    push_frame(vec178)

    # Keypoint layout: pose[0..24], left_hand[25..45], right_hand[46..66], face[67..88]
    hands_detected = bool(mask89[25:67].any())

    predictions = predict_top3() if hands_detected else []

    return JSONResponse({
        "predictions": predictions,
        "buffered_frames": buffered_frame_count(),
        "hands_detected": hands_detected,
        "landmarks": vec178.tolist(),   # 178 normalised floats
        "landmark_mask": mask89.tolist(),  # 89 bools, True = detected
    })


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "buffered_frames": buffered_frame_count()})


# Mount static files last — catches all unmatched routes and serves index.html
app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
