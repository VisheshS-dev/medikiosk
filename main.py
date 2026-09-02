"""
MediKiosk v2 — AI-Powered Clinical Intake Backend
100% Offline: Whisper (STT) + Ollama qwen2.5:3b (NLP) + llama3.2-vision (OCR)
"""

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Optional
import os, uuid, json, re, io, tempfile, base64, hashlib
from gtts import gTTS
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Intel Core Ultra 5 225H Hardware Optimizations ──
# Meteor Lake architecture features 4 Performance Cores (P-cores) + 8 Efficient Cores (E-cores).
# Pinning thread pools to the 4 physical P-cores avoids thread migration and the "E-core straggler"
# synchronization stall that degrades multi-threaded matrix operations on hybrid Intel CPUs.
INTEL_CPU_THREADS = int(os.getenv("INTEL_CPU_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(INTEL_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(INTEL_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(INTEL_CPU_THREADS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(INTEL_CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(INTEL_CPU_THREADS))
os.environ.setdefault("KMP_BLOCKTIME", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

load_dotenv()

# ── Offline AI Models ──
import ollama
import torch

if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(INTEL_CPU_THREADS)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(2)

def resolve_ollama_model():
    if os.getenv("OLLAMA_MODEL"):
        return os.getenv("OLLAMA_MODEL")
    try:
        res = ollama.list()
        names = [getattr(m, "model", str(m)) for m in getattr(res, "models", [])]
        # On mobile CPUs, qwen2.5:3b delivers 25-35 tok/s vs 6-8 tok/s on 7b.
        # Check for 3b first, then fallback to whatever is installed.
        for candidate in ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:latest"]:
            for name in names:
                if name.startswith(candidate):
                    return name
    except Exception:
        pass
    return "qwen2.5:7b"

OLLAMA_MODEL = resolve_ollama_model()
FALLBACK_MODEL = "qwen2.5:7b"
VISION_MODEL = os.getenv("VISION_MODEL", "moondream")
print(f"🏥 Active Ollama Model: {OLLAMA_MODEL} (Vision: {VISION_MODEL})")

import traceback

_whisper_model = None
_whisper_backend = None  # "faster_whisper" | "openai_whisper" | None

def load_whisper_model(preferred_gpu_model="large-v3", cpu_model="small"):
    global _whisper_model, _whisper_backend

    if _whisper_model is not None:
        return _whisper_model, _whisper_backend

    # 1. Try faster_whisper on CUDA if NVIDIA GPU is present
    try:
        from faster_whisper import WhisperModel
        if torch.cuda.is_available():
            try:
                print("Attempting to load faster-whisper (CUDA)...")
                model = WhisperModel(preferred_gpu_model, device="cuda", compute_type="int8_float16")
                _whisper_model = model
                _whisper_backend = "faster_whisper"
                print("✅ faster-whisper loaded on CUDA GPU.")
                return _whisper_model, _whisper_backend
            except Exception as e:
                print("faster_whisper CUDA load failed:", e)

        # 2. Intel Core Ultra CPU Optimization:
        # CTranslate2 utilizes AVX-VNNI instructions for native INT8 matrix acceleration on CPU.
        # Running with 4 threads matching physical P-cores avoids E-core latency penalties.
        try:
            print(f"Attempting to load faster-whisper on Intel CPU (model={cpu_model}, int8, {INTEL_CPU_THREADS} threads)...")
            model = WhisperModel(
                cpu_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=INTEL_CPU_THREADS,
                num_workers=1
            )
            _whisper_model = model
            _whisper_backend = "faster_whisper"
            print(f"✅ faster-whisper loaded on Intel CPU (AVX-VNNI int8, model={cpu_model}).")
            return _whisper_model, _whisper_backend
        except Exception as e:
            print("faster_whisper CPU load failed:", e)
    except Exception as e:
        print("faster_whisper not available:", e)

    # 3. Fallback: openai-whisper (PyTorch CPU, pinned to P-cores)
    try:
        print(f"Attempting to load openai-whisper (CPU) model: {cpu_model}")
        import whisper
        torch.set_num_threads(INTEL_CPU_THREADS)
        model = whisper.load_model(cpu_model, device="cpu")
        _whisper_model = model
        _whisper_backend = "openai_whisper"
        print(f"✅ openai-whisper loaded on CPU (model={cpu_model}).")
        return _whisper_model, _whisper_backend
    except Exception as e:
        print("openai-whisper failed to load:", e)
        traceback.print_exc()

    # Final fallback (no STT available)
    _whisper_model = None
    _whisper_backend = None
    print("❌ No whisper backend loaded. Speech-to-text will be unavailable.")
    return None, None


def transcribe_file(audio_path, model_name_for_cpu="small", **kwargs):
    """
    Transcribe audio_path using the available backend.
    Optimized for Intel Core Ultra CPU with greedy search (beam_size=1) and VAD filtering.
    """
    model, backend = load_whisper_model(cpu_model=model_name_for_cpu)
    if model is None:
        raise RuntimeError("No whisper model loaded; install faster-whisper or openai-whisper + dependencies.")

    if backend == "faster_whisper":
        transcribe_opts = {
            "beam_size": 1,
            "best_of": 1,
            "vad_filter": True,
            "vad_parameters": dict(min_silence_duration_ms=500, threshold=0.5),
            "condition_on_previous_text": False,
        }
        transcribe_opts.update(kwargs)
        segments, info = model.transcribe(audio_path, **transcribe_opts)
        return " ".join([seg.text for seg in segments]).strip()
    elif backend == "openai_whisper":
        result = model.transcribe(audio_path, **kwargs)
        return result.get("text", "").strip()
    else:
        raise RuntimeError("No whisper backend available.")


# ── LLM Helpers ──
def call_llm(prompt: str, image_bytes: Optional[bytes] = None, num_predict: Optional[int] = None) -> str:
    """Call Ollama with Intel P-core thread allocation, low-latency context, and optional token cap."""
    messages = [{'role': 'user', 'content': prompt}]
    model = OLLAMA_MODEL

    options = {
        'temperature': 0.1,
        'num_thread': INTEL_CPU_THREADS
    }
    if num_predict:
        options['num_predict'] = num_predict

    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        messages[0]['images'] = [b64]
        model = VISION_MODEL
        options['num_ctx'] = 2048
        print(f"→ Ollama Vision ({model})...")
        response = ollama.chat(model=model, messages=messages, format='json', options=options)
        return response['message']['content']

    options['num_ctx'] = 4096
    print(f"→ Ollama ({model})...")
    try:
        response = ollama.chat(model=model, messages=messages, format='json', options=options)
        return response['message']['content']
    except Exception as e:
        if model != FALLBACK_MODEL:
            print(f"⚠️ {model} not ready or failed ({e}), falling back to {FALLBACK_MODEL}...")
            response = ollama.chat(model=FALLBACK_MODEL, messages=messages, format='json', options=options)
            return response['message']['content']
        raise e


def extract_json_string(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def unwrap_json(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    keys = list(data.keys())
    if len(keys) == 1 and isinstance(data[keys[0]], dict):
        return data[keys[0]]
    if "properties" in data and isinstance(data["properties"], dict):
        return data["properties"]
    return data


# ── Language Codes ──
LANGUAGE_CODES = {
    "English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Bengali": "bn",
    "Gujarati": "gu", "Punjabi": "pa", "Urdu": "ur"
}

# ── Pre-written Follow-up Questions (human-verified, no AI typos) ──
PHASE_QUESTIONS = {
    "Hindi": {
        "initial": "यह तकलीफ़ आपको कब से हो रही है?",
        0: "क्या यह दर्द शरीर के किसी और हिस्से में भी जाता है?",
        1: "इसके साथ और कोई तकलीफ़ है? जैसे बुखार, उल्टी, या कमज़ोरी?",
        2: "क्या आपने इसके लिए कोई दवा ली है? किसी चीज़ से आराम मिलता है या तकलीफ़ बढ़ती है?",
        3: "क्या आपको पहले कोई बीमारी रही है? परिवार में किसी को कोई बीमारी है? क्या आप धूम्रपान या शराब का सेवन करते हैं?",
        4: "क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?",
        "default": "और कुछ बताना चाहेंगे?"
    },
    "English": {
        "initial": "How long have you been experiencing this problem?",
        0: "Does this pain spread or travel to any other part of your body?",
        1: "Are you experiencing any other symptoms like fever, nausea, or weakness?",
        2: "Have you taken any medicine for this? Does anything make it better or worse?",
        3: "Do you have any past medical conditions? Any diseases in your family? Do you smoke or drink alcohol?",
        4: "Are you allergic to any medicines or foods?",
        "default": "Is there anything else you would like to tell me?"
    },
    "Tamil": {
        "initial": "இந்த பிரச்சனை எவ்வளவு நாளாக இருக்கிறது?",
        0: "இந்த வலி உடலின் வேறு எந்த பகுதிக்கும் பரவுகிறதா?",
        1: "இதனுடன் காய்ச்சல், குமட்டல் அல்லது பலவீனம் போன்ற வேறு ஏதாவது தொந்தரவு இருக்கிறதா?",
        2: "இதற்கு ஏதாவது மருந்து எடுத்துக்கொண்டீர்களா? எதனால் சரியாகிறது அல்லது மோசமாகிறது?",
        3: "உங்களுக்கு முன்பு ஏதாவது நோய் இருந்ததா? குடும்பத்தில் யாருக்காவது நோய் இருக்கிறதா? புகைபிடிப்பீர்களா அல்லது மது அருந்துவீர்களா?",
        4: "உங்களுக்கு ஏதாவது மருந்து அல்லது உணவுக்கு ஒவ்வாமை இருக்கிறதா?",
        "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
    },
    "Telugu": {
        "initial": "ఈ సమస్య మీకు ఎంత కాలంగా ఉంది?",
        0: "ఈ నొప్పి శరీరంలో ఇతర భాగాలకు వ్యాపిస్తుందా?",
        1: "దీనితో పాటు జ్వరం, వాంతులు లేదా బలహీనత వంటి ఇతర సమస్యలు ఉన్నాయా?",
        2: "దీని కోసం ఏదైనా మందు వాడారా? దేనివల్ల తగ్గుతుంది లేదా పెరుగుతుంది?",
        3: "మీకు ఇంతకు ముందు ఏదైనా వ్యాధి ఉందా? కుటుంబంలో ఎవరికైనా వ్యాధి ఉందా? మీరు పొగ తాగుతారా లేదా మద్యం సేవిస్తారా?",
        4: "మీకు ఏదైనా మందు లేదా ఆహారానికి అలర్జీ ఉందా?",
        "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
    },
    "Bengali": {
        "initial": "এই সমস্যা আপনার কতদিন ধরে হচ্ছে?",
        0: "এই ব্যথা কি শরীরের অন্য কোনো জায়গায় ছড়ায়?",
        1: "এর সাথে জ্বর, বমি বা দুর্বলতার মতো অন্য কোনো সমস্যা আছে?",
        2: "এর জন্য কি কোনো ওষুধ খেয়েছেন? কিসে আরাম হয় বা কষ্ট বাড়ে?",
        3: "আগে কি কোনো রোগ ছিল? পরিবারে কারো কি কোনো রোগ আছে? আপনি কি ধূমপান বা মদ্যপান করেন?",
        4: "আপনার কি কোনো ওষুধ বা খাবারে অ্যালার্জি আছে?",
        "default": "আর কিছু বলতে চান?"
    },
    "Marathi": {
        "initial": "ही तकलीफ तुम्हाला कधीपासून होत आहे?",
        0: "हा दुखणे शरीराच्या इतर कोणत्या भागात जातो का?",
        1: "याबरोबर ताप, उलटी किंवा अशक्तपणा असे काही त्रास आहे का?",
        2: "यासाठी काही औषध घेतले का? कशामुळे आराम पडतो किंवा त्रास वाढतो?",
        3: "तुम्हाला आधी काही आजार होता का? कुटुंबात कोणाला काही आजार आहे का? तुम्ही धूम्रपान किंवा दारू पिता का?",
        4: "तुम्हाला कोणत्या औषधाची किंवा खाण्याच्या पदार्थाची ऍलर्जी आहे का?",
        "default": "अजून काही सांगायचे आहे का?"
    },
}

def get_phase_question(language: str, phase) -> str:
    """Get a pre-written question template. Falls back to English if language not available."""
    lang_questions = PHASE_QUESTIONS.get(language, PHASE_QUESTIONS.get("English", {}))
    return lang_questions.get(phase, lang_questions.get("default", "Is there anything else you would like to tell me?"))

# ── Pydantic Models ──
PATIENT_JSON_TEMPLATE = """{
  "chief_complaint": "Main presenting symptom (translated to clinical English)",
  "hpi": "Comprehensive narrative of present illness, onset, radiation, severity, associated symptoms, and relieving factors (in English)",
  "is_emergency": false,
  "severity": "Low|Medium|High",
  "duration": "Symptom duration (e.g. '2 days')",
  "past_medical_history": "Past conditions (or 'Uncertain / unconfirmed (patient does not recall)' if unsure, or 'Patient denies past chronic medical illness' if denied)",
  "family_history": "Family history (or 'Patient denies family history of similar illness' if denied, or 'Uncertain' if unsure)",
  "personal_history": "Smoking, alcohol, diet, habits (or 'No significant lifestyle risks reported')",
  "allergies": "Drug/food allergies (or 'No known drug allergies (NKDA)' if denied)",
  "review_of_systems": "Summary of systemic positive and negative findings (e.g. 'Patient reports diaphoresis and anxiety; denies fever, vomiting, or dyspnea')",
  "prakriti": "Not assessed",
  "vikriti": "Not assessed",
  "agni": "Not assessed",
  "next_question": "complete"
}"""

FOLLOWUP_JSON_TEMPLATE = """{
  "updates": {
    "hpi": null,
    "past_medical_history": null,
    "family_history": null,
    "personal_history": null,
    "allergies": null,
    "review_of_systems": null,
    "prakriti": null,
    "vikriti": null,
    "agni": null
  },
  "is_complete": false,
  "next_question": "Generated question in patient's language"
}"""

DOCUMENT_JSON_TEMPLATE = """{
  "document_type": "Name or type of report (e.g. CBC, MRI, Prescription)",
  "diagnoses": ["list of diagnoses"],
  "medications": ["medicine with dose"],
  "flagged_values": ["abnormal lab values"],
  "document_date": "date or Unknown",
  "summary": "Brief summary"
}"""

class PatientExtraction(BaseModel):
    chief_complaint: str
    hpi: Optional[str] = "None reported"
    is_emergency: bool = False
    severity: str = "Low"
    duration: str = "Unknown"
    past_medical_history: Optional[str] = "None reported"
    family_history: Optional[str] = "None reported"
    personal_history: Optional[str] = "None reported"
    allergies: Optional[str] = "None reported"
    review_of_systems: Optional[str] = "None reported"
    prakriti: Optional[str] = "Not assessed"
    vikriti: Optional[str] = "Not assessed"
    agni: Optional[str] = "Not assessed"
    next_question: Optional[str] = "Could you tell me more about this issue?"

class FollowUpResponse(BaseModel):
    updates: dict = {}
    is_complete: bool = False
    next_question: Optional[str] = "Is there anything else?"

class DocumentExtraction(BaseModel):
    document_type: str = "Medical Document"
    diagnoses: List[str] = []
    medications: List[str] = []
    flagged_values: List[str] = []
    document_date: str = "Unknown"
    summary: str = ""


# ── Database ──
SQLALCHEMY_DATABASE_URL = "sqlite:///./medikiosk_v2.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PatientRecord(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    abha_id = Column(String, index=True, nullable=True)  # Links visits by ABHA identity
    is_ayush = Column(Boolean, default=False)
    patient_name = Column(String, default="Patient")
    age = Column(String, default="")
    gender = Column(String, default="")
    phone = Column(String, default="")
    chief_complaint = Column(Text)
    hpi = Column(Text)
    is_emergency = Column(Boolean, default=False)
    severity = Column(String)
    duration = Column(String)
    past_medical_history = Column(String)
    family_history = Column(String)
    personal_history = Column(String)
    allergies = Column(String)
    review_of_systems = Column(Text)
    prakriti = Column(String)
    vikriti = Column(String)
    agni = Column(String)
    flagged_lab_values = Column(Text, default="[]")
    raw_dialogue = Column(Text, default="")
    is_synthesized = Column(Boolean, default=False)
    abha_relevance_json = Column(Text, default="{}")
    created_at = Column(String)


class VisitHistory(Base):
    """Stores past visit records linked by ABHA ID for history continuity."""
    __tablename__ = "visit_history"
    id = Column(Integer, primary_key=True, index=True)
    abha_id = Column(String, index=True)
    visit_date = Column(String)
    chief_complaint = Column(Text)
    diagnoses = Column(Text, default="[]")        # JSON list
    medications = Column(Text, default="[]")       # JSON list
    flagged_values = Column(Text, default="[]")    # JSON list
    summary = Column(Text)
    specialty = Column(String)
    is_relevant = Column(Boolean, default=False)   # Set by AI filter
    relevance_reason = Column(Text)                # Why AI thinks it's relevant


Base.metadata.create_all(bind=engine)

# Auto-migrate SQLite schema if columns don't exist
try:
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(patients)")).fetchall()]
        for col_name, col_type in [("abha_id", "VARCHAR"), ("patient_name", "VARCHAR"), ("age", "VARCHAR"), ("gender", "VARCHAR"), ("phone", "VARCHAR"), ("raw_dialogue", "TEXT"), ("is_synthesized", "BOOLEAN"), ("abha_relevance_json", "TEXT")]:
            if col_name not in cols:
                conn.execute(text(f"ALTER TABLE patients ADD COLUMN {col_name} {col_type}"))
        conn.commit()
except Exception as e:
    print(f"Auto-migration note: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── FastAPI App ──
app = FastAPI(title="MediKiosk v2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ── TTS (Text-to-Speech) Endpoint with Local Cache ──
TTS_CACHE_DIR = os.path.join("uploads", "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

TTS_LANG_MAP = {
    "Hindi": "hi", "English": "en", "Tamil": "ta", "Telugu": "te",
    "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn",
    "Malayalam": "ml", "Punjabi": "pa", "Urdu": "ur"
}

@app.get("/api/tts")
def get_tts(text: str, lang: str = "hi"):
    """Text-to-speech with local audio caching for 100% offline playback."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    lang_code = TTS_LANG_MAP.get(lang, lang.lower())
    if len(lang_code) > 2 and '-' in lang_code:
        lang_code = lang_code.split('-')[0]

    cache_key = hashlib.md5(f"{lang_code}_{text.strip()}".encode("utf-8")).hexdigest()
    cache_file = os.path.join(TTS_CACHE_DIR, f"{cache_key}.mp3")

    if not os.path.exists(cache_file):
        try:
            tts = gTTS(text=text.strip(), lang=lang_code)
            tts.save(cache_file)
        except Exception as e:
            print(f"TTS generation error: {e}")
            try:
                tts = gTTS(text=text.strip(), lang="en")
                tts.save(cache_file)
            except Exception as e2:
                print(f"TTS fallback error: {e2}")
                raise HTTPException(status_code=500, detail="TTS generation failed")

    return FileResponse(cache_file, media_type="audio/mpeg")



# ── 5 Diverse ABHA Patient Profiles (Mock ABDM Gateway) ──
ABHA_PROFILES = {
    "12-3456-7890-1234": {
        "name": "Ramesh Sharma",
        "age": 58,
        "gender": "Male",
        "phone": "9876543210",
        "abha_id": "12-3456-7890-1234",
        "avatar": "👨‍🦳",
        "badge": "Cardiology & Diabetes",
        "history": [
            {
                "visit_date": "2024-01-15",
                "chief_complaint": "Acute chest pain radiating to left arm with cold sweat",
                "diagnoses": json.dumps(["Myocardial Infarction (STEMI)", "Coronary Artery Disease"]),
                "medications": json.dumps(["Aspirin 75mg OD", "Clopidogrel 75mg OD", "Atorvastatin 40mg OD", "Metoprolol 25mg BD"]),
                "flagged_values": json.dumps(["Troponin I: 8.5 ng/mL (Critical High)", "CK-MB: 45 U/L (High)"]),
                "summary": "Admitted with acute STEMI. Underwent primary PCI with drug-eluting stent to LAD. Discharged on dual antiplatelet therapy.",
                "specialty": "Cardiology"
            },
            {
                "visit_date": "2024-06-20",
                "chief_complaint": "Follow-up cardiac & lipid evaluation",
                "diagnoses": json.dumps(["Hyperlipidemia", "Post-MI follow-up"]),
                "medications": json.dumps(["Atorvastatin 40mg OD", "Ramipril 5mg OD", "Aspirin 75mg OD"]),
                "flagged_values": json.dumps(["LDL Cholesterol: 145 mg/dL (High)", "Total Cholesterol: 230 mg/dL (High)"]),
                "summary": "6-month post-MI follow-up. Echocardiogram shows EF 50%. High LDL, statin dose maintained.",
                "specialty": "Cardiology"
            },
            {
                "visit_date": "2023-04-12",
                "chief_complaint": "Left ankle sprain after minor trip",
                "diagnoses": json.dumps(["Grade 1 Ankle Sprain"]),
                "medications": json.dumps(["Paracetamol 650mg SOS", "Diclofenac gel"]),
                "flagged_values": json.dumps([]),
                "summary": "Minor ligament strain. Fully resolved in 2 weeks.",
                "specialty": "Orthopedics"
            },
            {
                "visit_date": "2025-02-10",
                "chief_complaint": "Fatigue, increased thirst, and frequent urination",
                "diagnoses": json.dumps(["Type 2 Diabetes Mellitus (Uncontrolled)", "Hypertension Stage 2"]),
                "medications": json.dumps(["Metformin 500mg BD", "Glimepiride 1mg OD", "Telmisartan 40mg OD"]),
                "flagged_values": json.dumps(["HbA1c: 8.2% (High)", "Fasting Blood Glucose: 168 mg/dL (High)", "BP: 152/94 mmHg"]),
                "summary": "Uncontrolled Type 2 Diabetes with Stage 2 HTN. Oral hypoglycemics adjusted.",
                "specialty": "General Medicine"
            }
        ]
    },
    "23-4567-8901-2345": {
        "name": "Priya Patel",
        "age": 32,
        "gender": "Female",
        "phone": "9812345678",
        "abha_id": "23-4567-8901-2345",
        "avatar": "👩",
        "badge": "Pulmonology & Asthma",
        "history": [
            {
                "visit_date": "2024-10-05",
                "chief_complaint": "Severe acute breathlessness, dry cough, and wheezing",
                "diagnoses": json.dumps(["Acute Exacerbation of Bronchial Asthma", "Bronchospasm"]),
                "medications": json.dumps(["Salbutamol Nebulization SOS", "Budecort Inhaler 200mcg BD", "Montelukast 10mg OD"]),
                "flagged_values": json.dumps(["Serum IgE: 650 IU/mL (Markedly Elevated)", "Peak Expiratory Flow: 220 L/min (Low)"]),
                "summary": "Severe asthma attack triggered by dust and cold weather. Responsive to bronchodilators.",
                "specialty": "Pulmonology"
            },
            {
                "visit_date": "2023-02-18",
                "chief_complaint": "Facial skin acne breakouts",
                "diagnoses": json.dumps(["Acne Vulgaris"]),
                "medications": json.dumps(["Clindamycin 1% gel", "Benzoyl Peroxide 2.5%"]),
                "flagged_values": json.dumps([]),
                "summary": "Mild papular acne treated with topical antibiotics.",
                "specialty": "Dermatology"
            },
            {
                "visit_date": "2024-03-14",
                "chief_complaint": "Persistent nighttime coughing fits",
                "diagnoses": json.dumps(["Cough-Variant Asthma", "Allergic Bronchitis"]),
                "medications": json.dumps(["Levocetirizine 5mg OD", "Formoterol + Budesonide Inhaler"]),
                "flagged_values": json.dumps([]),
                "summary": "Nocturnal asthma symptoms controlled with combination inhaler therapy.",
                "specialty": "Pulmonology"
            }
        ]
    },
    "34-5678-9012-3456": {
        "name": "Sunita Devi",
        "age": 64,
        "gender": "Female",
        "phone": "9765432109",
        "abha_id": "34-5678-9012-3456",
        "avatar": "👵",
        "badge": "Orthopedics & Arthritis",
        "history": [
            {
                "visit_date": "2024-08-11",
                "chief_complaint": "Severe right knee joint pain, crepitus, and inability to climb stairs",
                "diagnoses": json.dumps(["Primary Osteoarthritis of Right Knee (Grade 3)", "Synovial Effusion"]),
                "medications": json.dumps(["Aceclofenac 100mg + Paracetamol 325mg BD", "Glucosamine 1500mg OD"]),
                "flagged_values": json.dumps(["Knee X-Ray: Medial compartment joint space narrowing with subchondral sclerosis"]),
                "summary": "Advanced knee osteoarthritis. Advised physiotherapy, quadriceps strengthening, and knee brace.",
                "specialty": "Orthopedics"
            },
            {
                "visit_date": "2023-11-20",
                "chief_complaint": "Diffuse lower back pain and bone aches",
                "diagnoses": json.dumps(["Osteopenia", "Severe Vitamin D3 Deficiency"]),
                "medications": json.dumps(["Cholecalciferol (Vit D3) 60,000 IU weekly", "Calcium Carbonate 500mg BD"]),
                "flagged_values": json.dumps(["Serum Vitamin D: 9.8 ng/mL (Deficient)", "DEXA T-Score: -2.1 (Osteopenia)"]),
                "summary": "Osteopenia identified on DEXA scan. Intensive Vitamin D and Calcium supplementation started.",
                "specialty": "Rheumatology"
            },
            {
                "visit_date": "2022-05-04",
                "chief_complaint": "Difficulty reading small text and driving at night",
                "diagnoses": json.dumps(["Early Immature Senile Cataract", "Presbyopia"]),
                "medications": json.dumps(["Carboxymethylcellulose eye drops"]),
                "flagged_values": json.dumps([]),
                "summary": "Prescription glasses updated. Annual eye review advised.",
                "specialty": "Ophthalmology"
            }
        ]
    },
    "45-6789-0123-4567": {
        "name": "Mohammed Ali",
        "age": 45,
        "gender": "Male",
        "phone": "9654321098",
        "abha_id": "45-6789-0123-4567",
        "avatar": "👨",
        "badge": "Gastroenterology & Liver",
        "history": [
            {
                "visit_date": "2024-04-22",
                "chief_complaint": "Burning epigastric pain, acid regurgitation, and post-meal fullness",
                "diagnoses": json.dumps(["Erosive Reflux Esophagitis (Grade B)", "Antral Gastritis"]),
                "medications": json.dumps(["Pantoprazole 40mg OD", "Sucralfate suspension 10ml TDS"]),
                "flagged_values": json.dumps(["Endoscopy: Multiple superficial linear mucosal erosions in lower third of esophagus"]),
                "summary": "Endoscopy confirmed erosive GERD. 8-week course of PPI and mucosal protectant prescribed.",
                "specialty": "Gastroenterology"
            },
            {
                "visit_date": "2024-11-19",
                "chief_complaint": "Dull ache in right upper quadrant of abdomen and mild nausea",
                "diagnoses": json.dumps(["Non-Alcoholic Fatty Liver Disease (Grade 1 NAFLD)", "Elevated Liver Enzymes"]),
                "medications": json.dumps(["Ursodeoxycholic Acid (UDCA) 300mg BD", "Vitamin E 400mg OD"]),
                "flagged_values": json.dumps(["SGPT/ALT: 68 U/L (High)", "SGOT/AST: 54 U/L (High)", "Serum Bilirubin: 1.1 mg/dL"]),
                "summary": "Abdominal ultrasound showed Grade 1 hepatic steatosis with elevated transaminases.",
                "specialty": "Gastroenterology"
            },
            {
                "visit_date": "2023-07-08",
                "chief_complaint": "Right ear itching and discomfort after swimming",
                "diagnoses": json.dumps(["Acute Otitis Externa"]),
                "medications": json.dumps(["Ciprofloxacin ear drops", "Ibuprofen 400mg"]),
                "flagged_values": json.dumps([]),
                "summary": "Swimmer's ear cleared completely after 5 days of topical antibiotic drops.",
                "specialty": "ENT"
            }
        ]
    },
    "56-7890-1234-5678": {
        "name": "Anita Verma",
        "age": 26,
        "gender": "Female",
        "phone": "9543210987",
        "abha_id": "56-7890-1234-5678",
        "avatar": "👩‍🦰",
        "badge": "ENT & Allergy",
        "history": [
            {
                "visit_date": "2024-12-01",
                "chief_complaint": "Bilateral throbbing facial pressure, thick nasal discharge, and frontal headache",
                "diagnoses": json.dumps(["Acute Exacerbation of Chronic Maxillary Sinusitis"]),
                "medications": json.dumps(["Amoxicillin-Clavulanate 625mg BD x 7d", "Fluticasone Furoate Nasal Spray 1 puff BD", "Saline rinse"]),
                "flagged_values": json.dumps(["PNS X-Ray: Bilateral maxillary sinus haziness with mucosal thickening"]),
                "summary": "Bacterial sinusitis flare-up treated with oral antibiotics and steroid nasal spray.",
                "specialty": "ENT"
            },
            {
                "visit_date": "2023-09-15",
                "chief_complaint": "Excessive morning sneezing bouts (15-20 sneezes) and watery itchy eyes",
                "diagnoses": json.dumps(["Perennial Allergic Rhinitis (Dust Mite Allergy)"]),
                "medications": json.dumps(["Bilastine 20mg OD", "Montelukast 10mg OD"]),
                "flagged_values": json.dumps(["Skin Prick Test: Positive for Dermatophagoides pteronyssinus (House Dust Mite)"]),
                "summary": "Allergen sensitization confirmed. Prescribed non-sedating antihistamines and dust avoidance measures.",
                "specialty": "Allergy / Immunology"
            },
            {
                "visit_date": "2022-01-10",
                "chief_complaint": "Right wrist tenderness from repetitive typing",
                "diagnoses": json.dumps(["De Quervain's Tenosynovitis (Right Wrist)"]),
                "medications": json.dumps(["Thumb spica splint", "Diclofenac gel"]),
                "flagged_values": json.dumps([]),
                "summary": "Repetitive strain injury. Advised ergonomic workstation setup and rest.",
                "specialty": "Orthopedics"
            }
        ]
    }
}


def seed_abha_history(abha_id: str, db: Session):
    """Seed mock ABHA history into VisitHistory table if not already present."""
    if not abha_id:
        return
    # Check if already seeded for this specific abha_id
    existing = db.query(VisitHistory).filter(VisitHistory.abha_id == abha_id).first()
    if existing:
        return
    
    profile = ABHA_PROFILES.get(abha_id)
    if not profile or "history" not in profile:
        return
    
    for visit in profile["history"]:
        record = VisitHistory(
            abha_id=abha_id,
            visit_date=visit["visit_date"],
            chief_complaint=visit["chief_complaint"],
            diagnoses=visit["diagnoses"],
            medications=visit["medications"],
            flagged_values=visit["flagged_values"],
            summary=visit["summary"],
            specialty=visit["specialty"],
            is_relevant=False,
            relevance_reason=None
        )
        db.add(record)
    db.commit()
    print(f"✅ Seeded {len(profile['history'])} mock ABHA history records for {profile['name']} ({abha_id})")


# ── Context-Aware History Filter (Background Task) ──
HISTORY_FILTER_TEMPLATE = """{
  "results": [
    {"visit_id": 1, "is_relevant": true, "reason": "Detailed clinical correlation explaining why this past record is directly relevant to today's complaint"},
    {"visit_id": 2, "is_relevant": false, "reason": "Clinical explanation why this past visit is unrelated"}
  ]
}"""


def filter_history_background(patient_id_db: int, abha_id: str, chief_complaint: str):
    """Background task: Uses Qwen to filter past ABHA history strictly by relevance to THIS patient's complaint."""
    db = SessionLocal()
    try:
        patient = db.query(PatientRecord).filter(PatientRecord.id == patient_id_db).first()
        past_visits = db.query(VisitHistory).filter(VisitHistory.abha_id == abha_id).all()
        if not past_visits or not patient:
            return
        
        # Build concise history summary for LLM
        visits_for_llm = []
        for v in past_visits:
            visits_for_llm.append({
                "visit_id": v.id,
                "date": v.visit_date,
                "complaint": v.chief_complaint,
                "diagnoses": json.loads(v.diagnoses) if v.diagnoses else [],
                "medications": json.loads(v.medications) if v.medications else [],
                "flagged_values": json.loads(v.flagged_values) if v.flagged_values else [],
                "specialty": v.specialty
            })
        
        prompt = f"""You are a clinical decision support AI acting on behalf of a doctor reviewing a patient's historical medical records.
The patient is presenting TODAY at the triage kiosk with the following Chief Complaint:
"{chief_complaint}"

Here is the patient's verified past medical history from ABHA:
{json.dumps(visits_for_llm, indent=2)}

TASK: For EACH past visit, determine if it is MEDICALLY RELEVANT to today's chief complaint ("{chief_complaint}").

CLINICAL RELEVANCE RULES:
1. RELEVANT (is_relevant: true):
   - Involves the SAME organ system, anatomical region, or related etiology (e.g. past STEMI/Heart Attack or HTN is RELEVANT when patient has chest pain or breathlessness; Gastroenterology/Gastritis is RELEVANT when presenting for abdominal pain).
   - Involves active medications that could interact or explain current symptoms.
   - Contains lab flags or chronic diagnoses directly tied to the current complaint.
2. NOT RELEVANT (is_relevant: false):
   - Belongs to a completely unrelated specialty/organ system (e.g., Cardiology/Heart Attack or Knee Osteoarthritis when presenting for Abdominal pain).
   - A minor, completely resolved past issue with no clinical bearing on today's presentation.

IMPORTANT: Set `is_relevant` to true or false. Provide a concise, professional clinical reason in `reason`.

Output ONLY valid JSON:
{HISTORY_FILTER_TEMPLATE}"""

        print(f"→ Running ABHA clinical relevance filter for '{chief_complaint}' on patient {patient.patient_id}...")
        response_text = call_llm(prompt, num_predict=512)
        result_json = json.loads(extract_json_string(response_text))
        result_json = unwrap_json(result_json)
        
        results = result_json.get("results", [])
        if not isinstance(results, list):
            results = [result_json] if isinstance(result_json, dict) else []
        
        relevance_map = {}
        for item in results:
            visit_id = item.get("visit_id")
            raw_rel = item.get("is_relevant")
            if isinstance(raw_rel, bool):
                is_relevant = raw_rel
            else:
                is_relevant = str(raw_rel).strip().lower() in ["true", "1", "yes"]
            reason = str(item.get("reason", "")).strip()
            if visit_id is not None:
                relevance_map[str(visit_id)] = {
                    "is_relevant": is_relevant,
                    "reason": reason
                }
        
        patient.abha_relevance_json = json.dumps(relevance_map)
        db.commit()
        print(f"✅ ABHA clinical relevance filter complete for patient {patient.patient_id}. {sum(1 for r in relevance_map.values() if r.get('is_relevant'))} relevant visits attached.")
    except Exception as e:
        print(f"❌ History filter error: {e}")
    finally:
        db.close()


# ── Unsure vs Denial Helpers ──
UNSURE_PATTERNS = [
    "not sure", "unsure", "dont know", "don't know", "not certain", "uncertain", 
    "no idea", "pata nahi", "pata nahin", "malum nahi", "maloom nahi", "nahi pata", 
    "yaad nahi", "yaad nhi", "yaad nahin", "याद नहीं", "याद नाही", "confirm nahi", 
    "confirm nhi", "not confirmed", "unconfirmed", "bhul gaya", "bhool gaya", 
    "mai confirm nahi", "main confirm nahi", "mujhe yaad nahi", "mujhe yaad nhi", 
    "mujhe confirm nahi", "mujhe confirm nhi", "shayad", "maybe", "not remembered",
    "पता नहीं", "मालूम नहीं", "माहित नाही", "తెలియదు", "தெரியாது", "জানা নেই", 
    "ખબર નથી", "ಗೊತ್ತಿಲ್ಲ", "ಅറിയിಲ್ಲ", "ਨਹੀਂ ਪਤਾ", "🤷 not sure", "🤷"
]

DENIAL_PATTERNS = [
    "no", "nahi", "nahin", "नहीं", "no allergies", "none", "nothing", 
    "kuch nahi", "kuch nahi hai", "na", "न", "ना", "n", "nope", "never", "nil", "✓ no", "✗ no"
]

def is_unsure_response(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(p in t for p in UNSURE_PATTERNS)

def is_denial_response(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if is_unsure_response(t):
        return False
    if t in DENIAL_PATTERNS:
        return True
    return any(t == p or t.startswith(p + " ") or t.endswith(" " + p) for p in DENIAL_PATTERNS)


# ── Instant Patient Builder (0ms LLM Overhead) ──
def build_patient_from_transcript(transcript, language, is_ayush, pt_id, db, abha_id=None, patient_name="Patient", age="", gender="", phone=""):
    t_lower = transcript.lower()
    is_emerg = any(w in t_lower for w in [
        "chest pain", "heart", "attack", "breath", "stroke", "paralysis", "unconscious", "bleeding", "accident",
        "सीने में दर्द", "हार्ट", "अटैक", "सांस", "बेहोश", "खून", "छातीत दुखणे", "గుండె", "நெஞ்சு வலி", "বুকের ব্যথা"
    ])

    dialogue_entry = f"Patient (Chief Complaint - {language}): {transcript}\n"

    patient = PatientRecord(
        patient_id=pt_id,
        abha_id=abha_id,
        is_ayush=bool(is_ayush),
        patient_name=patient_name or "Patient",
        age=str(age) if age else "",
        gender=str(gender) if gender else "",
        phone=str(phone) if phone else "",
        chief_complaint=transcript,
        hpi=f"Patient reports: {transcript}",
        is_emergency=is_emerg,
        severity="High" if is_emerg else "Medium",
        duration="Recording in progress",
        past_medical_history="Awaiting synthesis",
        family_history="Awaiting synthesis",
        personal_history="Awaiting synthesis",
        allergies="Awaiting synthesis",
        review_of_systems="Awaiting synthesis",
        prakriti="Not assessed",
        vikriti="Not assessed",
        agni="Not assessed",
        raw_dialogue=dialogue_entry,
        is_synthesized=False,
        created_at=datetime.now().strftime("%I:%M %p")
    )
    if db is not None:
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # Seed mock ABHA history if available
    if abha_id and db is not None:
        seed_abha_history(abha_id, db)

    # Use SOCRATES-framework question template
    initial_question = get_phase_question(language, "initial")
    return patient, initial_question


# ── Stage 2: Post-Interview Holistic Clinical Synthesis (Background Task) ──
def synthesize_and_filter_patient_background(patient_id_db: int, abha_id: Optional[str], language: str, is_ayush: bool):
    """Runs after intake/document scan finishes: synthesizes full consultation + uploaded documents and filters ABHA records."""
    db = SessionLocal()
    try:
        patient = db.query(PatientRecord).filter(PatientRecord.id == patient_id_db).first()
        if not patient:
            return
        
        full_transcript = patient.raw_dialogue or patient.chief_complaint
        ayush_inst = (
            "The setting is Ayurvedic OPD. Assess Prakriti, Vikriti, Agni if evident."
            if is_ayush else
            "The setting is standard Allopathic. Set prakriti, vikriti, agni to 'Not assessed'."
        )

        docs_summary_str = ""
        if patient.flagged_lab_values and patient.flagged_lab_values != "[]":
            try:
                docs_list = json.loads(patient.flagged_lab_values)
                if isinstance(docs_list, list) and len(docs_list) > 0:
                    docs_summary_str = "\n\nUploaded Clinical Documents / Lab Reports:\n"
                    for idx, d in enumerate(docs_list, 1):
                        if isinstance(d, dict):
                            docs_summary_str += f"- Doc #{idx} ({d.get('document_type', 'Report')} - Date: {d.get('document_date', 'Unknown')}): Summary: {d.get('summary', '')}, Diagnoses: {d.get('diagnoses', [])}, Meds: {d.get('medications', [])}, Flagged Labs: {d.get('flagged_values', [])}\n"
            except Exception as e:
                print(f"Error parsing docs for synthesis: {e}")

        prompt = f"""You are an expert Chief Medical Officer and AI Clinical Scribe.
Review the complete patient consultation dialogue and all uploaded medical documents below.

Language spoken: {language}
{ayush_inst}

Complete Consultation Dialogue:
{full_transcript}
{docs_summary_str}

TASK: Perform high-precision clinical synthesis into a standard medical English EHR record.

CRITICAL RULES FOR CLINICAL ACCURACY:
1. ACCURATELY DISTINGUISH DENIAL ("NO") VS UNCERTAINTY ("NOT SURE / DON'T REMEMBER") VS NOT ASKED:
   - When the patient clearly DENIES or says "No" / "नहीं" / "कुछ नहीं" / "नहीं है":
     * family_history: Write "Patient denies family history of similar complaints / No significant family history". NEVER write "Not reported" when the patient explicitly denied it!
     * past_medical_history: Write "Patient denies past chronic medical conditions / No significant past medical history".
     * allergies: Write "No known drug or food allergies (NKDA)".
     * personal_history: Write "No significant lifestyle or habit risks reported".
   - When the patient expresses UNCERTAINTY or LACK OF MEMORY (e.g., "yaad nahi", "confirm nahi", "pata nahi", "not sure", "don't remember", "uncertain"):
     * Record clearly as "Uncertain / unconfirmed (patient does not recall / unsure)". DO NOT write "No" or "Denies"!
   - When a category was NOT asked in dialogue or documents:
     * Record as "Not assessed".

2. REVIEW OF SYSTEMS (ROS):
   - Actively summarize all associated systemic symptoms asked or reported during the interview (e.g. "Patient denies fever, vomiting, or dyspnea; reports diaphoresis and nausea" or "Patient denies associated systemic symptoms"). DO NOT leave empty or as "None reported".

3. INTEGRATE UPLOADED DOCUMENTS & LAB REPORTS:
   - If uploaded documents/reports are present above, integrate their diagnoses, lab flags, and findings into HPI, past history, and review of systems.

4. EMERGENCY TRIAGE & SEVERITY:
   - Set `is_emergency`: true if red flags (acute coronary syndrome, stroke signs, severe trauma, acute respiratory distress), else false.
   - Set `severity`: "High" | "Medium" | "Low".

5. Set `next_question` to 'complete'.

Output ONLY valid JSON:
{PATIENT_JSON_TEMPLATE}"""

        print(f"→ Synthesizing full clinical record (dialogue + documents) for patient {patient.patient_id} in background...")
        response_text = call_llm(prompt, num_predict=600)
        result_json = json.loads(extract_json_string(response_text))
        result_json = unwrap_json(result_json)
        ext = PatientExtraction(**result_json)

        # Update patient record with synthesized clinical data
        patient.chief_complaint = ext.chief_complaint or patient.chief_complaint
        patient.hpi = ext.hpi or patient.hpi
        patient.is_emergency = ext.is_emergency
        patient.severity = ext.severity or patient.severity
        patient.duration = ext.duration or "Unknown"
        patient.past_medical_history = ext.past_medical_history or "No significant past medical history"
        patient.family_history = ext.family_history or "No significant family history"
        patient.personal_history = ext.personal_history or "No significant lifestyle risks"
        patient.allergies = ext.allergies or "No known drug allergies (NKDA)"
        patient.review_of_systems = ext.review_of_systems or "Patient denies associated systemic symptoms"
        patient.prakriti = ext.prakriti if is_ayush else "Not assessed"
        patient.vikriti = ext.vikriti if is_ayush else "Not assessed"
        patient.agni = ext.agni if is_ayush else "Not assessed"
        patient.is_synthesized = True

        db.commit()
        print(f"✅ Clinical record synthesis complete for {patient.patient_id} ({patient.chief_complaint})")

        # Now correlate and filter past ABHA visit records against the full synthesized clinical profile
        if abha_id:
            filter_history_background(patient_id_db, abha_id, patient.chief_complaint)

    except Exception as e:
        print(f"❌ Synthesis error: {e}")
    finally:
        db.close()


# ═══════════════ ENDPOINTS ═══════════════

@app.get("/")
async def root():
    return {"message": "MediKiosk v2 Backend running (Offline Mode)"}


# ── ABHA Master Profiles Endpoint ──
@app.get("/api/abha-profiles")
async def get_abha_profiles():
    """Returns the 5 pre-configured ABHA patient profiles for quick lookup."""
    return [
        {
            "name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "phone": p["phone"],
            "abha_id": p["abha_id"],
            "avatar": p["avatar"],
            "badge": p["badge"],
            "history_count": len(p["history"])
        }
        for p in ABHA_PROFILES.values()
    ]

@app.get("/api/abha-profile/{abha_id}")
async def get_abha_profile_by_id(abha_id: str):
    """Lookup demographic details for a given ABHA ID with hyphen/space normalization."""
    raw_clean = re.sub(r'[\s\-]', '', abha_id.strip())
    
    for key, p in ABHA_PROFILES.items():
        key_clean = re.sub(r'[\s\-]', '', key)
        if raw_clean == key_clean or abha_id.strip().lower() == key.lower():
            return {
                "found": True,
                "name": p["name"],
                "age": p["age"],
                "gender": p["gender"],
                "phone": p["phone"],
                "abha_id": p["abha_id"],
                "badge": p["badge"],
                "summary": p["summary"] if "summary" in p else ""
            }
    return {
        "found": False,
        "message": f"No ABHA account found with number '{abha_id}'. Please check the 14-digit number and try again."
    }


# ── Initial Complaint (Audio) ──
@app.post("/api/process-audio")
async def process_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    language: str = Form("English"),
    is_ayush: bool = Form(False),
    abha_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form("Patient"),
    age: Optional[str] = Form(""),
    gender: Optional[str] = Form(""),
    phone: Optional[str] = Form(""),
    db: Session = Depends(get_db)
):
    audio_bytes = await audio.read()
    pt_id = f"PT-{str(uuid.uuid4())[:4].upper()}"

    # 1. Whisper STT
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    lang_code = LANGUAGE_CODES.get(language, "en")
    transcript = transcribe_file(
        tmp_path,
        model_name_for_cpu="small",
        language=lang_code,
        initial_prompt="A clinical consultation in a hospital OPD. Symptoms, pain, fever, duration, past history."
    )
    os.remove(tmp_path)
    print(f"Transcript: {transcript}")

    # 2. Instant patient initialization (0ms LLM calls)
    patient, next_q = build_patient_from_transcript(
        transcript=transcript,
        language=language,
        is_ayush=is_ayush,
        pt_id=pt_id,
        db=db,
        abha_id=abha_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        phone=phone
    )

    return {
        "status": "success",
        "extracted_complaint": patient.chief_complaint,
        "is_emergency": patient.is_emergency,
        "patient_id": patient.patient_id,
        "patient_name": patient.patient_name,
        "abha_id": patient.abha_id,
        "transcript": transcript,
        "next_question": next_q
    }


# ── Initial Complaint (Text) ──
@app.post("/api/process-text")
async def process_text(
    background_tasks: BackgroundTasks,
    transcript: str = Form(...),
    language: str = Form("English"),
    is_ayush: bool = Form(False),
    abha_id: Optional[str] = Form(None),
    patient_name: Optional[str] = Form("Patient"),
    age: Optional[str] = Form(""),
    gender: Optional[str] = Form(""),
    phone: Optional[str] = Form(""),
    db: Session = Depends(get_db)
):
    pt_id = f"PT-{str(uuid.uuid4())[:4].upper()}"
    patient, next_q = build_patient_from_transcript(
        transcript=transcript,
        language=language,
        is_ayush=is_ayush,
        pt_id=pt_id,
        db=db,
        abha_id=abha_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        phone=phone
    )

    return {
        "status": "success",
        "extracted_complaint": patient.chief_complaint,
        "is_emergency": patient.is_emergency,
        "patient_id": patient.patient_id,
        "patient_name": patient.patient_name,
        "abha_id": patient.abha_id,
        "transcript": transcript,
        "next_question": next_q
    }


def handle_followup_extraction(
    patient: PatientRecord,
    transcript: str,
    language: str,
    conversation_context: str,
    is_ayush: bool,
    follow_up_count: int,
    db: Session,
    background_tasks: Optional[BackgroundTasks] = None
):
    phase_names = {
        0: "Duration / Onset",
        1: "Radiation of Pain",
        2: "Associated Symptoms (ROS)",
        3: "Medications & Relieving Factors",
        4: "Past Medical / Family / Lifestyle",
        5: "Allergies"
    }
    phase_label = phase_names.get(follow_up_count, f"Phase {follow_up_count}")
    prev_q = get_phase_question(language, follow_up_count - 1 if follow_up_count > 0 else "initial")
    dialogue_entry = f"Doctor ({phase_label}): {prev_q}\nPatient ({language}): {transcript}\n"

    current_dialogue = patient.raw_dialogue or ""
    patient.raw_dialogue = current_dialogue + dialogue_entry

    is_complete = True if follow_up_count >= 5 else False
    next_question = get_phase_question(language, follow_up_count) if follow_up_count < 5 else "Thank you. Let us proceed to document scanning."

    # When the 5-question interview finishes, trigger full AI synthesis & ABHA filter in background
    if is_complete and background_tasks is not None:
        print(f"🚀 Patient intake complete! Dispatching holistic AI synthesis & ABHA history filter for {patient.patient_id}...")
        background_tasks.add_task(synthesize_and_filter_patient_background, patient.id, patient.abha_id, language, is_ayush)

    if db is not None:
        db.commit()
        db.refresh(patient)

    return {
        "status": "success",
        "extracted_info": f"Recorded: {transcript}",
        "is_complete": is_complete,
        "next_question": next_question,
        "transcript": transcript
    }


# ── Follow-up (Text) ──
@app.post("/api/follow-up-text")
async def follow_up_text(
    background_tasks: BackgroundTasks,
    transcript: str = Form(...),
    language: str = Form("English"),
    patient_id: str = Form(...),
    conversation_context: str = Form(""),
    is_ayush: bool = Form(False),
    follow_up_count: int = Form(0),
    db: Session = Depends(get_db)
):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return handle_followup_extraction(
        patient=patient,
        transcript=transcript,
        language=language,
        conversation_context=conversation_context,
        is_ayush=is_ayush,
        follow_up_count=follow_up_count,
        db=db,
        background_tasks=background_tasks
    )


# ── Follow-up (Audio) ──
@app.post("/api/follow-up")
async def follow_up_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    language: str = Form("English"),
    patient_id: str = Form(...),
    conversation_context: str = Form(""),
    is_ayush: bool = Form(False),
    follow_up_count: int = Form(0),
    db: Session = Depends(get_db)
):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    audio_bytes = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    lang_code = LANGUAGE_CODES.get(language, "en")
    
     # Use your custom wrapper that handles the CPU fallback
    transcript = transcribe_file(
        tmp_path,
        model_name_for_cpu="small",
        language=lang_code,
        initial_prompt="A clinical consultation in a hospital OPD. Symptoms, pain, fever, duration, past history."
    )
    os.remove(tmp_path)
    print(f"Follow-up transcript: {transcript}")

    return handle_followup_extraction(
        patient=patient,
        transcript=transcript,
        language=language,
        conversation_context=conversation_context,
        is_ayush=is_ayush,
        follow_up_count=follow_up_count,
        db=db,
        background_tasks=background_tasks
    )


def process_document_background(file_bytes: bytes, filename: str, content_type: str, file_url: str, patient_id_db: int):
    db = SessionLocal()
    try:
        patient = db.query(PatientRecord).filter(PatientRecord.id == patient_id_db).first()
        if not patient:
            return
            
        structured_data = None
        extracted_text = ""
        try:
            prompt_base = """Analyze this medical document carefully.

Extract into JSON:
- document_type: Name or type of report (e.g. CBC, MRI, Prescription)
- diagnoses: list of clinical diagnoses (English)
- medications: list of medicines with dosages (English)
- flagged_values: list of abnormal lab values or critical findings
- document_date: date on document, or 'Unknown'
- summary: concise summary of key findings

Output ONLY valid JSON:
""" + DOCUMENT_JSON_TEMPLATE

            if filename.lower().endswith('.pdf') or content_type == 'application/pdf':
                import fitz
                pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in pdf_doc:
                    extracted_text += page.get_text() + "\n"
                prompt = f"Extracted Text:\n{extracted_text}\n\n{prompt_base}"
                response_text = call_llm(prompt)
            else:
                print(f"Using {VISION_MODEL} for image in background...")
                response_text = call_llm(prompt_base, image_bytes=file_bytes)

            result_json = json.loads(extract_json_string(response_text))
            result_json = unwrap_json(result_json)
            extraction = DocumentExtraction(**result_json)
            structured_data = {
                "document_type": extraction.document_type,
                "diagnoses": extraction.diagnoses,
                "medications": extraction.medications,
                "flagged_values": extraction.flagged_values,
                "document_date": extraction.document_date,
                "summary": extraction.summary,
                "file_url": file_url,
                "raw_text": extracted_text
            }
            print(f"Background Extracted: {structured_data}")
        except Exception as e:
            print(f"Background Document processing failed: {e}")

        if not structured_data:
            structured_data = {
                "document_type": "Unknown Document",
                "diagnoses": ["Extraction failed — please try again"],
                "medications": [],
                "flagged_values": [],
                "document_date": "Unknown",
                "summary": "Could not process this document. Please try a clearer image.",
                "file_url": file_url,
                "raw_text": extracted_text
            }

        existing = []
        if patient.flagged_lab_values and patient.flagged_lab_values != "[]":
            try:
                parsed = json.loads(patient.flagged_lab_values)
                if isinstance(parsed, list):
                    existing = [i for i in parsed if isinstance(i, dict)]
            except:
                pass
        existing.append(structured_data)
        patient.flagged_lab_values = json.dumps(existing)
        db.commit()
    finally:
        db.close()


# ── Document Processing ──
@app.post("/api/process-document")
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    patient = None
    if patient_id:
        patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        patient = db.query(PatientRecord).order_by(PatientRecord.id.desc()).first()
    if not patient:
        raise HTTPException(status_code=404, detail="No patient found")

    file_bytes = await file.read()
    print(f"Document received: {file.filename}, {len(file_bytes)} bytes. Dispatching background task.")

    # Save file for viewing later
    os.makedirs("uploads", exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or '.png'
    saved_filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join("uploads", saved_filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    
    # We will pass the full url, assuming frontend is on same host or API is absolute
    base_url = "http://localhost:8000" 
    file_url = f"{base_url}/uploads/{saved_filename}"

    # Dispatch to background task
    background_tasks.add_task(
        process_document_background,
        file_bytes,
        file.filename,
        file.content_type,
        file_url,
        patient.id
    )

    return {
        "status": "success", 
        "message": "Document is being processed asynchronously.",
        "extracted_document": {
            "document_type": "Processing...",
            "diagnoses": ["Analyzing document in background..."],
            "medications": [],
            "flagged_values": [],
            "document_date": "Pending",
            "summary": "Document securely uploaded and queued for processing.",
            "file_url": file_url,
            "raw_text": ""
        }
    }


# ── Finalize Intake & Trigger Comprehensive Synthesis (Spoken Dialogue + Documents) ──
@app.post("/api/finalize-intake")
async def finalize_intake(
    background_tasks: BackgroundTasks,
    patient_id: str = Form(...),
    language: str = Form("English"),
    is_ayush: bool = Form(False),
    db: Session = Depends(get_db)
):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    background_tasks.add_task(synthesize_and_filter_patient_background, patient.id, patient.abha_id, language, is_ayush)
    return {"status": "success", "message": "Comprehensive synthesis (dialogue + documents) queued"}


# ── Red Flag Check ──
@app.get("/api/red-flag-check")
async def red_flag_check(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        return {"has_red_flags": False, "flags": [], "message": "Patient not found"}

    # Fast path: If emergency was already detected during intake keywords, return immediately (0ms latency)
    if patient.is_emergency:
        return {
            "has_red_flags": True,
            "flags": [patient.chief_complaint or "Potential urgent symptoms"],
            "message": "Immediate clinical attention recommended."
        }

    prompt = f"""You are a medical triage safety system. Based on the patient's reported symptoms, identify any RED FLAG symptoms that may require urgent clinical assessment.

Chief Complaint: {patient.chief_complaint}
HPI: {patient.hpi}
Severity: {patient.severity}

Respond with JSON:
{{"has_red_flags": true/false, "flags": ["list of concerning symptoms"], "message": "brief explanation"}}

IMPORTANT: Do NOT diagnose. Only flag potentially urgent symptoms. Be conservative — flag if uncertain."""

    try:
        response_text = call_llm(prompt, num_predict=128)
        result = json.loads(extract_json_string(response_text))
        result = unwrap_json(result)
        return result
    except:
        return {"has_red_flags": False, "flags": [], "message": "Safety check completed — no urgent flags detected."}


# ── Specialty Matching ──
@app.get("/api/specialty-match")
async def specialty_match(patient_id: str, language: str = "English", db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        return {"specialty": "General Medicine", "reason": "Default", "confidence": "Low"}

    prompt = f"""Based on the patient's reported symptoms, suggest the most appropriate medical specialty.

Chief Complaint: {patient.chief_complaint}
HPI: {patient.hpi}

Respond with JSON:
{{"specialty": "Specialty name in {language}", "reason": "Brief explanation in {language}", "confidence": "Low|Medium|High"}}

Common specialties: General Medicine, Cardiology, Pulmonology, Gastroenterology, Neurology, Orthopedics, Dermatology, ENT, Ophthalmology, Psychiatry, Obstetrics & Gynecology, Pediatrics, Urology, Surgery.

IMPORTANT: The JSON keys must remain in English, but the VALUES for 'specialty' and 'reason' MUST be accurately translated into {language}. Do NOT diagnose. Only suggest which specialty is most appropriate for the described symptoms."""

    try:
        response_text = call_llm(prompt, num_predict=128)
        result = json.loads(extract_json_string(response_text))
        result = unwrap_json(result)
        return result
    except:
        return {"specialty": "General Medicine", "reason": "Default recommendation", "confidence": "Medium"}


# ── Patient Queue ──
@app.get("/api/patients")
async def get_patients(db: Session = Depends(get_db)):
    patients = db.query(PatientRecord).order_by(PatientRecord.id.desc()).all()
    return [{
        "patient_id": p.patient_id,
        "patient_name": p.patient_name or "Patient",
        "age": p.age,
        "gender": p.gender,
        "abha_id": p.abha_id,
        "created_at": p.created_at,
        "is_emergency": p.is_emergency
    } for p in patients]

@app.delete("/api/patients/{patient_id}")
async def delete_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    db.delete(patient)
    db.commit()
    return {"status": "success", "message": f"Patient {patient_id} deleted"}


# ── Patient Summary ──
@app.get("/api/patient-summary")
async def get_patient_summary(patient_id: Optional[str] = None, db: Session = Depends(get_db)):
    if patient_id:
        patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    else:
        patient = db.query(PatientRecord).order_by(PatientRecord.id.desc()).first()

    if not patient:
        return {"status": "No patients yet"}

    return {
        "patient_id": patient.patient_id,
        "patient_name": patient.patient_name or "Patient",
        "age": patient.age or "",
        "gender": patient.gender or "",
        "phone": patient.phone or "",
        "abha_id": patient.abha_id,
        "chief_complaint": patient.chief_complaint or "Not recorded",
        "hpi": patient.hpi or "None reported",
        "is_emergency": patient.is_emergency,
        "severity": patient.severity or "Unknown",
        "duration": patient.duration or "Unknown",
        "past_medical_history": patient.past_medical_history or "None reported",
        "family_history": patient.family_history or "None reported",
        "personal_history": patient.personal_history or "None reported",
        "allergies": patient.allergies or "None reported",
        "review_of_systems": patient.review_of_systems or "None reported",
        "prakriti": patient.prakriti or "Not assessed",
        "vikriti": patient.vikriti or "Not assessed",
        "agni": patient.agni or "Not assessed",
        "flagged_lab_values": patient.flagged_lab_values or "[]",
        "created_at": patient.created_at,
    }


# ── Patient History (ABHA-linked past visits) ──
@app.get("/api/patient-history")
async def get_patient_history(patient_id: str, db: Session = Depends(get_db)):
    """Returns past visit history strictly for this patient's linked ABHA profile."""
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if not patient or not patient.abha_id:
        return {"relevant_history": [], "other_history": [], "filter_status": "no_abha", "abha_id": None}
    
    # Ensure profile history is seeded if not present
    seed_abha_history(patient.abha_id, db)
    
    past_visits = db.query(VisitHistory).filter(VisitHistory.abha_id == patient.abha_id).all()
    if not past_visits:
        return {"relevant_history": [], "other_history": [], "filter_status": "no_history", "abha_id": patient.abha_id}
    
    # Load this patient's isolated relevance mapping
    relevance_map = {}
    if patient.abha_relevance_json:
        try:
            relevance_map = json.loads(patient.abha_relevance_json)
        except Exception:
            relevance_map = {}
    
    filter_complete = bool(relevance_map)
    
    relevant = []
    other = []
    for v in past_visits:
        rel_info = relevance_map.get(str(v.id), {})
        is_rel = rel_info.get("is_relevant", False)
        reason = rel_info.get("reason", "")
        
        visit_data = {
            "id": v.id,
            "visit_date": v.visit_date,
            "chief_complaint": v.chief_complaint,
            "diagnoses": json.loads(v.diagnoses) if v.diagnoses else [],
            "medications": json.loads(v.medications) if v.medications else [],
            "flagged_values": json.loads(v.flagged_values) if v.flagged_values else [],
            "summary": v.summary,
            "specialty": v.specialty,
            "is_relevant": is_rel,
            "relevance_reason": reason
        }
        if is_rel:
            relevant.append(visit_data)
        else:
            other.append(visit_data)
    
    # Sort by date descending
    relevant.sort(key=lambda x: x["visit_date"], reverse=True)
    other.sort(key=lambda x: x["visit_date"], reverse=True)
    
    return {
        "relevant_history": relevant,
        "other_history": other,
        "filter_status": "complete" if filter_complete else "processing",
        "abha_id": patient.abha_id,
        "patient_name": patient.patient_name
    }


@app.post("/api/demo-data")
async def demo_data(background_tasks: BackgroundTasks, abha_id: Optional[str] = "12-3456-7890-1234", db: Session = Depends(get_db)):
    profile = ABHA_PROFILES.get(abha_id, ABHA_PROFILES["12-3456-7890-1234"])
    pt_id = f"PT-DEMO-{str(uuid.uuid4())[:4].upper()}"
    
    demo_doc = {
        "document_type": "Lipid Profile & ECG Report",
        "diagnoses": ["Hyperlipidemia", "CAD Status Post-PCI"],
        "medications": ["Atorvastatin 40mg OD", "Aspirin 75mg OD"],
        "flagged_values": ["LDL Cholesterol: 145 mg/dL (High)", "Total Cholesterol: 230 mg/dL (High)"],
        "document_date": datetime.now().strftime("%Y-%m-%d"),
        "summary": "Follow-up lab report showing elevated LDL cholesterol and stable cardiac rhythm.",
        "file_url": "",
        "raw_text": "Demo cardiology follow-up document"
    }

    patient = PatientRecord(
        patient_id=pt_id,
        abha_id=profile["abha_id"],
        patient_name=profile["name"],
        age=str(profile["age"]),
        gender=profile["gender"],
        phone=profile["phone"],
        chief_complaint="Severe retrosternal chest pain with left arm radiation and sweating",
        hpi="• Started 2 hours ago while walking\n• Crushing substernal pressure, severity 8/10\n• Accompanied by diaphoresis and mild nausea",
        is_emergency=True,
        severity="High",
        duration="2 hours",
        past_medical_history="• Myocardial Infarction in 2024 (Stented LAD)\n• Type 2 Diabetes Mellitus",
        family_history="• Father had premature CAD at age 52",
        personal_history="• Non-smoker, vegetarian diet",
        allergies="• None reported",
        review_of_systems="• No fever\n• Shortness of breath on exertion",
        prakriti="Not assessed",
        vikriti="Not assessed",
        agni="Not assessed",
        flagged_lab_values=json.dumps([demo_doc]),
        created_at=datetime.now().strftime("%I:%M %p")
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    # Seed ABHA history and trigger AI filter in background
    seed_abha_history(profile["abha_id"], db)
    background_tasks.add_task(filter_history_background, patient.id, profile["abha_id"], patient.chief_complaint)

    return {"status": "success", "patient_id": pt_id, "patient_name": profile["name"], "abha_id": profile["abha_id"]}


if __name__ == "__main__":
    print("🏥 Starting MediKiosk v2 Backend (Offline Mode)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
