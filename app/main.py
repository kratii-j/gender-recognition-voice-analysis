from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.audio import classify_voice, load_wav_mono
from app.storage import HistoryStore

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "app.db"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Gender Recognition Using Voice", version="3.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
store = HistoryStore(DB_PATH)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/api/history")
def history():
    return store.list_recent()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "recording.wav").suffix.lower()
    if suffix != ".wav":
        raise HTTPException(status_code=400, detail="Please upload a WAV file.")

    audio_id = uuid4().hex
    target = AUDIO_DIR / f"{audio_id}.wav"
    content = await file.read()
    target.write_bytes(content)

    try:
        signal, sample_rate = load_wav_mono(target)
        result = classify_voice(signal, sample_rate)
    except Exception as error:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error)) from error

    record = {
        "id": audio_id,
        "filename": file.filename or target.name,
        **result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    store.insert(record)
    return record
