import os
import subprocess
import tempfile
import time
from typing import List, Optional
import psutil
import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI(title="Local AI Workstation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = "qwen2.5:7b"

# Cached Whisper model
whisper_model_cache = {}

def get_whisper_model(model_size: str = "small"):
    if model_size not in whisper_model_cache:
        from faster_whisper import WhisperModel
        try:
            # First try GPU float16 on RTX 3060
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"CUDA initialization fallback to CPU/int8: {e}")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        whisper_model_cache[model_size] = model
    return whisper_model_cache[model_size]


def get_gpu_info():
    """Extract GPU name and VRAM usage via nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "available": True,
            "name": parts[0],
            "vram_used_mb": int(parts[1]),
            "vram_total_mb": int(parts[2]),
            "temp_c": int(parts[3]),
            "gpu_util_percent": int(parts[4]),
        }
    except Exception:
        return {
            "available": False,
            "name": "NVIDIA GPU",
            "vram_used_mb": 0,
            "vram_total_mb": 12288,
            "temp_c": 0,
            "gpu_util_percent": 0,
        }


@app.get("/api/status")
def status():
    # Check Ollama
    ollama_ok = False
    installed_models = []
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if r.status_code == 200:
            ollama_ok = True
            data = r.json()
            installed_models = [m["name"] for m in data.get("models", [])]
    except Exception:
        ollama_ok = False

    # RAM & CPU
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    gpu_info = get_gpu_info()

    return {
        "status": "online",
        "ollama_connected": ollama_ok,
        "models": installed_models,
        "active_model": DEFAULT_MODEL,
        "system": {
            "cpu_percent": cpu_percent,
            "ram_used_gb": round((mem.total - mem.available) / (1024**3), 1),
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "ram_percent": mem.percent,
            "gpu": gpu_info,
        },
    }


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = DEFAULT_MODEL
    stream: Optional[bool] = True


@app.post("/api/chat")
def chat(req: ChatRequest):
    payload = {
        "model": req.model or DEFAULT_MODEL,
        "messages": [m.dict() for m in req.messages],
        "stream": req.stream,
    }

    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=req.stream, timeout=120)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {r.text}")

        if not req.stream:
            return r.json()

        def stream_generator():
            for line in r.iter_lines():
                if line:
                    yield line.decode("utf-8") + "\n"

        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Не удалось подключиться к Ollama. Убедитесь, что служба запущена ({e})",
        )


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Form("small"),
    language: Optional[str] = Form(None),
):
    """Transcribes audio/video using GPU-accelerated faster-whisper."""
    start_time = time.time()
    suffix = os.path.splitext(file.filename)[1] or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        whisper = get_whisper_model(model_size)
        lang_arg = language if language and language != "auto" else None
        
        segments, info = whisper.transcribe(
            tmp_path,
            language=lang_arg,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        results = []
        full_text_parts = []
        for segment in segments:
            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            })
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)
        elapsed = round(time.time() - start_time, 2)

        return {
            "filename": file.filename,
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration": round(info.duration, 2),
            "processing_time_sec": elapsed,
            "speedup_factor": round(info.duration / elapsed, 1) if elapsed > 0 else 0,
            "full_text": full_text,
            "segments": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка транскрибации: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/api/analyze-doc")
async def analyze_doc(
    file: UploadFile = File(...),
    prompt: str = Form("Сделай подробное структурированное резюме этого документа и выдели ключевые пункты."),
    model: str = Form(DEFAULT_MODEL),
):
    """Extracts text from PDF/TXT and asks Ollama to analyze it."""
    ext = os.path.splitext(file.filename)[1].lower()
    content = await file.read()
    extracted_text = ""

    if ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            tmp.write(content)
        try:
            reader = PdfReader(tmp_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(f"--- Страница {i+1} ---\n{text}")
            extracted_text = "\n\n".join(pages_text)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # Assume text / utf-8
        try:
            extracted_text = content.decode("utf-8")
        except UnicodeDecodeError:
            extracted_text = content.decode("cp1251", errors="ignore")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Не удалось извлечь текст из файла.")

    # Limit text to ~25,000 characters to keep context clean
    trimmed_text = extracted_text[:25000]
    if len(extracted_text) > 25000:
        trimmed_text += "\n\n[Документ был обрезан до первых 25 000 символов]"

    system_instruction = (
        "Ты — профессиональный аналитик документов. Внимательно изучи предоставленный текст документа "
        "и ответь на запрос пользователя четко, структурированно, на русском языке."
    )

    user_query = f"ТЕКСТ ДОКУМЕНТА:\n\"\"\"\n{trimmed_text}\n\"\"\"\n\nЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{prompt}"

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model or DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_query},
                ],
                "stream": False,
            },
            timeout=180,
        )
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {r.text}")
        data = r.json()
        analysis = data.get("message", {}).get("content", "")
        return {
            "filename": file.filename,
            "characters_analyzed": len(trimmed_text),
            "analysis": analysis,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа документа: {str(e)}")


# Mount static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
