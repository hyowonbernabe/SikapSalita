"""
Sikap-Salita live demo FastAPI server.

Start with:
    cd scratch/PANSINAYAN
    .venv/Scripts/python -m uvicorn live_demo.app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from live_demo.keypoints import extract_from_base64, get_models, shutdown_models
from live_demo.inference import push_frame, predict_top3, buffered_frame_count
from live_demo.model import load_model
from live_demo.labels import load_labels
from live_demo.tts import get_audio, warm_cache

_STATIC = Path(__file__).parent / "static"

# 105-label English → Filipino override dictionary (matches labels_reference.csv order)
_FIL: dict[str, str] = {
    # GREETING
    "GOOD MORNING":       "Magandang umaga",
    "GOOD AFTERNOON":     "Magandang hapon",
    "GOOD EVENING":       "Magandang gabi",
    "HELLO":              "Kumusta",
    "HOW ARE YOU":        "Kamusta ka",
    "IM FINE":            "Mabuti naman ako",
    "NICE TO MEET YOU":   "Ikinagagalak kitang makilala",
    "THANK YOU":          "Salamat",
    "YOURE WELCOME":      "Walang anuman",
    "SEE YOU TOMORROW":   "Hanggang bukas",
    # SURVIVAL
    "UNDERSTAND":         "Naiintindihan",
    "DON'T UNDERSTAND":   "Hindi naiintindihan",
    "KNOW":               "Alam",
    "DON'T KNOW":         "Hindi ko alam",
    "NO":                 "Hindi",
    "YES":                "Oo",
    "WRONG":              "Mali",
    "CORRECT":            "Tama",
    "SLOW":               "Mabagal",
    "FAST":               "Mabilis",
    # NUMBER
    "ONE":                "Isa",
    "TWO":                "Dalawa",
    "THREE":              "Tatlo",
    "FOUR":               "Apat",
    "FIVE":               "Lima",
    "SIX":                "Anim",
    "SEVEN":              "Pito",
    "EIGHT":              "Walo",
    "NINE":               "Siyam",
    "TEN":                "Sampu",
    # CALENDAR
    "JANUARY":            "Enero",
    "FEBRUARY":           "Pebrero",
    "MARCH":              "Marso",
    "APRIL":              "Abril",
    "MAY":                "Mayo",
    "JUNE":               "Hunyo",
    "JULY":               "Hulyo",
    "AUGUST":             "Agosto",
    "SEPTEMBER":          "Setyembre",
    "OCTOBER":            "Oktubre",
    "NOVEMBER":           "Nobyembre",
    "DECEMBER":           "Disyembre",
    # DAYS
    "MONDAY":             "Lunes",
    "TUESDAY":            "Martes",
    "WEDNESDAY":          "Miyerkules",
    "THURSDAY":           "Huwebes",
    "FRIDAY":             "Biyernes",
    "SATURDAY":           "Sabado",
    "SUNDAY":             "Linggo",
    "TODAY":              "Ngayon",
    "TOMORROW":           "Bukas",
    "YESTERDAY":          "Kahapon",
    # FAMILY
    "FATHER":             "Tatay",
    "MOTHER":             "Nanay",
    "SON":                "Anak na lalaki",
    "DAUGHTER":           "Anak na babae",
    "GRANDFATHER":        "Lolo",
    "GRANDMOTHER":        "Lola",
    "UNCLE":              "Tito",
    "AUNTIE":             "Tita",
    "COUSIN":             "Pinsan",
    "PARENTS":            "Magulang",
    # RELATIONSHIPS
    "BOY":                "Batang lalaki",
    "GIRL":               "Batang babae",
    "MAN":                "Lalaki",
    "WOMAN":              "Babae",
    "DEAF":               "Bingi",
    "HARD OF HEARING":    "Mahirap marinig",
    "WEELCHAIR PERSON":   "Taong gumagamit ng wheelchair",
    "BLIND":              "Bulag",
    "DEAF BLIND":         "Bingi at bulag",
    "MARRIED":            "Kasal",
    # COLOR
    "BLUE":               "Asul",
    "GREEN":              "Berde",
    "RED":                "Pula",
    "BROWN":              "Kayumanggi",
    "BLACK":              "Itim",
    "WHITE":              "Puti",
    "YELLOW":             "Dilaw",
    "ORANGE":             "Kahel",
    "GRAY":               "Kulay abo",
    "PINK":               "Rosas",
    "VIOLET":             "Lila",
    "LIGHT":              "Maliwanag",
    "DARK":               "Madilim",
    # FOOD
    "BREAD":              "Tinapay",
    "EGG":                "Itlog",
    "FISH":               "Isda",
    "MEAT":               "Karne",
    "CHICKEN":            "Manok",
    "SPAGHETTI":          "Spaghetti",
    "RICE":               "Kanin",
    "LONGANISA":          "Longganisa",
    "SHRIMP":             "Hipon",
    "CRAB":               "Alimango",
    # DRINK
    "HOT":                "Mainit",
    "COLD":               "Malamig",
    "JUICE":              "Juice",
    "MILK":               "Gatas",
    "COFFEE":             "Kape",
    "TEA":                "Tsaa",
    "BEER":               "Beer",
    "WINE":               "Alak",
    "SUGAR":              "Asukal",
    "NO SUGAR":           "Walang asukal",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    get_models()
    labels = load_labels()

    # Pre-warm TTS cache for all 105 labels in both languages
    # (runs in background — demo works even if this is still in progress)
    print("[Sikap-Salita] Pre-warming TTS cache…")
    en_labels = [v["label"] for v in labels.values()]
    fil_labels = [_FIL.get(lbl, lbl) for lbl in en_labels]
    import asyncio
    asyncio.create_task(warm_cache(en_labels, fil_labels))

    print("[Sikap-Salita] Ready — open http://localhost:8000")
    yield
    shutdown_models()


app = FastAPI(title="Sikap-Salita Live Demo", lifespan=lifespan)


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    frame: str  # base64 JPEG, with or without data-URI prefix


@app.post("/predict")
async def predict(req: PredictRequest) -> JSONResponse:
    vec178, mask89 = extract_from_base64(req.frame)
    push_frame(vec178)

    hands_detected = bool(mask89[25:67].any())
    predictions = predict_top3() if hands_detected else []

    return JSONResponse({
        "predictions": predictions,
        "buffered_frames": buffered_frame_count(),
        "hands_detected": hands_detected,
        "landmarks": vec178.tolist(),
        "landmark_mask": mask89.tolist(),
    })


# ── /tts ──────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str        # English FSL label or pre-translated Filipino text
    lang: str = "en" # "en" | "fil"


@app.post("/tts")
async def tts(req: TTSRequest) -> Response:
    """Return MP3 audio for *text* in *lang*.

    For Filipino, translates English FSL labels via the override dictionary
    before synthesis so callers can always pass the raw English label.
    """
    text = req.text.strip()
    lang = req.lang if req.lang in ("en", "fil") else "en"

    if lang == "fil":
        text = _FIL.get(text.upper(), text)

    try:
        audio = await get_audio(text, lang)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"TTS error: {exc}")

    return Response(content=audio, media_type="audio/mpeg")


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "buffered_frames": buffered_frame_count()})


# Static files last — catches all unmatched routes
app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
