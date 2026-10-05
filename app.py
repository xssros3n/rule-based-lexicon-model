"""
FastAPI moderation inference service.
"""
import os
import json
import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List

from moderation.service import ModerationService

app = FastAPI(title="Moderation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(Path(__file__).parent, "static")), name="static")

from fastapi.responses import FileResponse, RedirectResponse

@app.get("/")
def read_root():
    # Redirect directly to the Swagger UI for API testing
    return RedirectResponse(url="/docs")

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent / "model.joblib"))
LEXICON_PATH = os.path.join(Path(__file__).parent, "profanity_lexicon.json")
SAFE_LEXICON_PATH = os.path.join(Path(__file__).parent, "safe_lexicon.json")
DB_PATH = os.path.join(Path(__file__).parent, "db.json")

service = None

def load_db():
    if not os.path.exists(DB_PATH):
        return {"warnings": [], "reports": []}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(data):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@app.on_event("startup")
async def load_model():
    global service
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    service = ModerationService(MODEL_PATH, lexicon_path=LEXICON_PATH, safe_lexicon_path=SAFE_LEXICON_PATH)
    
    # Initialize DB if not exists
    if not os.path.exists(DB_PATH):
        save_db({"warnings": [], "reports": []})

class ModerateRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    message_id: Optional[str] = None

class ModerateResponse(BaseModel):
    allowed: bool
    censored_text: str
    is_abusive: bool
    action: str
    reason_code: str | None = None
    detected_spans: list = []

@app.post("/moderate", response_model=ModerateResponse)
def moderate(req: ModerateRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    if service is None:
        raise HTTPException(status_code=503, detail="Service not loaded")

    result = service.moderate(req.text)
    
    # Log warning if abusive
    if result.get("is_abusive") and req.user_id:
        db = load_db()
        db["warnings"].append({
            "user_id": req.user_id,
            "message_id": req.message_id,
            "original_text": req.text,
            "detected_spans": result["detected_spans"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        save_db(db)
        
    return ModerateResponse(**result)

class ReportRequest(BaseModel):
    message_id: str
    text: str
    reported_by: str
    sender_id: str

@app.post("/report")
def report_message(req: ReportRequest):
    # Analyze the reported message
    result = service.moderate(req.text)
    
    db = load_db()
    report_entry = {
        "message_id": req.message_id,
        "reported_by": req.reported_by,
        "sender_id": req.sender_id,
        "text": req.text,
        "is_abusive": result["is_abusive"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    db["reports"].append(report_entry)
    
    # If abusive, we also log a warning for the sender
    if result["is_abusive"]:
        db["warnings"].append({
            "user_id": req.sender_id,
            "message_id": req.message_id,
            "original_text": req.text,
            "detected_spans": result["detected_spans"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source": "user_report"
        })
    save_db(db)
    
    return {
        "status": "success", 
        "is_abusive": result["is_abusive"],
        "message": "Report logged. User warned." if result["is_abusive"] else "Report logged. Message was safe."
    }

class AddWordRequest(BaseModel):
    word: str
    severity: str = "high"

@app.post("/admin/add_word")
def add_word(req: AddWordRequest):
    if not req.word or not req.word.strip():
        raise HTTPException(status_code=400, detail="Empty word")
    
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            lexicon = json.load(f)
            
        word_clean = req.word.strip().lower()
        if word_clean in lexicon:
            return {"status": "already_exists", "word": word_clean}
            
        lexicon[word_clean] = req.severity
        
        with open(LEXICON_PATH, "w", encoding="utf-8") as f:
            json.dump(lexicon, f, indent=2, ensure_ascii=False)
            
        if service and hasattr(service, 'span_detector'):
            from moderation.span_detector import SpanDetector
            service.span_detector = SpanDetector(lexicon_path=LEXICON_PATH)
            
        return {"status": "success", "message": f"Word '{word_clean}' added instantly!", "word": word_clean}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RemoveWordRequest(BaseModel):
    word: str

@app.post("/admin/remove_word")
def remove_word(req: RemoveWordRequest):
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            lexicon = json.load(f)
            
        word_clean = req.word.strip().lower()
        if word_clean not in lexicon:
            return {"status": "not_found", "message": "Word not in blocklist."}
            
        del lexicon[word_clean]
        
        with open(LEXICON_PATH, "w", encoding="utf-8") as f:
            json.dump(lexicon, f, indent=2, ensure_ascii=False)
            
        if service and hasattr(service, 'span_detector'):
            from moderation.span_detector import SpanDetector
            service.span_detector = SpanDetector(lexicon_path=LEXICON_PATH)
            
        return {"status": "success", "message": f"Word '{word_clean}' removed instantly!", "word": word_clean}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/get_words")
def get_words():
    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            lexicon = json.load(f)
        return {"words": lexicon}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AddSafeWordRequest(BaseModel):
    word: str

@app.post("/admin/add_safe_word")
def add_safe_word(req: AddSafeWordRequest):
    if not req.word or not req.word.strip():
        raise HTTPException(status_code=400, detail="Empty word")
    
    try:
        safe_lexicon = []
        if os.path.exists(SAFE_LEXICON_PATH):
            with open(SAFE_LEXICON_PATH, "r", encoding="utf-8") as f:
                safe_lexicon = json.load(f)
                
        word_clean = req.word.strip().lower()
        if word_clean in safe_lexicon:
            return {"status": "already_exists", "word": word_clean}
            
        safe_lexicon.append(word_clean)
        
        with open(SAFE_LEXICON_PATH, "w", encoding="utf-8") as f:
            json.dump(safe_lexicon, f, indent=2, ensure_ascii=False)
            
        if service:
            service.safe_lexicon = safe_lexicon
            
        return {"status": "success", "message": f"Word '{word_clean}' added to allowlist instantly!", "word": word_clean}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RemoveSafeWordRequest(BaseModel):
    word: str

@app.post("/admin/remove_safe_word")
def remove_safe_word(req: RemoveSafeWordRequest):
    try:
        safe_lexicon = []
        if os.path.exists(SAFE_LEXICON_PATH):
            with open(SAFE_LEXICON_PATH, "r", encoding="utf-8") as f:
                safe_lexicon = json.load(f)
                
        word_clean = req.word.strip().lower()
        if word_clean not in safe_lexicon:
            return {"status": "not_found", "message": "Word not in allowlist."}
            
        safe_lexicon.remove(word_clean)
        
        with open(SAFE_LEXICON_PATH, "w", encoding="utf-8") as f:
            json.dump(safe_lexicon, f, indent=2, ensure_ascii=False)
            
        if service:
            service.safe_lexicon = safe_lexicon
            
        return {"status": "success", "message": f"Word '{word_clean}' removed from allowlist instantly!", "word": word_clean}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/get_safe_words")
def get_safe_words():
    try:
        if os.path.exists(SAFE_LEXICON_PATH):
            with open(SAFE_LEXICON_PATH, "r", encoding="utf-8") as f:
                safe_lexicon = json.load(f)
            return {"safe_words": safe_lexicon}
        return {"safe_words": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/warnings")
def get_warnings():
    return {"warnings": load_db()["warnings"]}

@app.get("/admin/reports")
def get_reports():
    return {"reports": load_db()["reports"]}

@app.get("/health")
def health():
    return {"status": "ok", "service_loaded": service is not None}
