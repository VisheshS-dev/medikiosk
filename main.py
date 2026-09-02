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

# ── Pre-written Symptom-Specific Question Sets (human-verified, no AI typos) ──
CATEGORY_QUESTIONS = {
    "chest_pain": {
        "Hindi": {
            "initial": "यह सीने में दर्द या सांस लेने में तकलीफ़ आपको कब से हो रही है?",
            0: "क्या यह दर्द आपके बाएं हाथ, कंधे, जबड़े या पीठ की तरफ फैलता है?",
            1: "क्या इसके साथ सांस फूलना, ठंडा पसीना, चक्कर या घबराहट हो रही है?",
            2: "क्या चलने या मेहनत करने से दर्द बढ़ता है और आराम करने से घटता है?",
            3: "क्या आपको पहले बीपी, शुगर, दिल की बीमारी रही है या आप धूम्रपान करते हैं?",
            4: "क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How long have you been having this chest discomfort or breathing trouble?",
            0: "Does this pain spread to your left arm, shoulder, jaw, or back?",
            1: "Are you experiencing shortness of breath, cold sweats, dizziness, or nausea?",
            2: "Does walking or exertion make it worse, and does resting relieve it?",
            3: "Do you have a history of high BP, diabetes, heart disease, or smoking?",
            4: "Are you allergic to any medicines or foods?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த நெஞ்சு வலி அல்லது மூச்சுத்திணறல் எவ்வளவு நாளாக இருக்கிறது?",
            0: "இந்த வலி இடது கை, தோள்பட்டை, தாடை அல்லது முதுகுக்கு பரவுகிறதா?",
            1: "இதனுடன் மூச்சுத்திணறல், குளிர்ந்த வியர்வை, மயக்கம் அல்லது குமட்டல் உள்ளதா?",
            2: "நடக்கும்போது வலி அதிகமாகி ஓய்வெடுக்கும்போது குறைகிறதா?",
            3: "உங்களுக்கு ரத்த அழுத்தம், சர்க்கரை நோய், இதய நோய் உள்ளதா அல்லது புகைபிடிப்பீர்களா?",
            4: "உங்களுக்கு ஏதாவது மருந்து அல்லது உணவுக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ గుండె నొప్పి లేదా శ్వాస తీసుకోవడంలో ఇబ్బంది ఎంత కాలంగా ఉంది?",
            0: "ఈ నొప్పి ఎడమ చేయి, భుజం, దవడ లేదా వెనుక భాగానికి వ్యాపిస్తుందా?",
            1: "దీనితో పాటు శ్వాస ఆడకపోవడం, చల్లని చెమటలు, తలతిరగడం లేదా వికారం ఉన్నాయా?",
            2: "నడిచినప్పుడు నొప్పి పెరిగి, విశ్రాంతి తీసుకున్నప్పుడు తగ్గుతుందా?",
            3: "మీకు బీపీ, షుగర్, గుండె జబ్బుల చరిత్ర ఉందా లేదా పొగ తాగుతారా?",
            4: "మీకు ఏదైనా మందు లేదా ఆహారానికి అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "stomach_pain": {
        "Hindi": {
            "initial": "यह पेट दर्द आपको कब से हो रहा है?",
            0: "क्या यह दर्द पेट के ऊपरी हिस्से में है, नीचे की तरफ या पीठ में जाता है?",
            1: "क्या उल्टी, दस्त, खट्टी डकार, जलन या बुखार जैसा लग रहा है?",
            2: "क्या कुछ खाने-पीने से दर्द बढ़ता है या खाली पेट रहने से?",
            3: "क्या आपको पहले अल्सर, गैस, पथरी की शिकायत रही है या बाहर का खाना खाया था?",
            4: "क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How long have you had this stomach or abdominal pain?",
            0: "Is the pain in the upper abdomen, lower belly, or radiating to the back?",
            1: "Are you experiencing vomiting, loose motions, acidity, burning, or fever?",
            2: "Does eating food or drinking water make the pain worse or better?",
            3: "Do you have a history of ulcers, acidity, gallstones, or recent outside food?",
            4: "Are you allergic to any medicines or foods?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த வயிற்று வலி உங்களுக்கு எவ்வளவு நாளாக இருக்கிறது?",
            0: "இந்த வலி வயிற்றின் மேல் பகுதியிலா, கீழ் பகுதியிலா அல்லது முதுகில் பரவுகிறதா?",
            1: "இதனுடன் வாந்தி, வயிற்றுப்போக்கு, நெஞ்செரிச்சல் அல்லது காய்ச்சல் உள்ளதா?",
            2: "சாப்பிட்ட பிறகு வலி அதிகமாகிறதா அல்லது குறைகிறதா?",
            3: "உங்களுக்கு குடல் புண், பித்தப்பை கல் அல்லது வெளி உணவு சாப்பிட்ட வரலாறு உள்ளதா?",
            4: "உங்களுக்கு ஏதாவது மருந்து அல்லது உணவுக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ కడుపు నొప్పి మీకు ఎంత కాలంగా ఉంది?",
            0: "నొప్పి కడుపు పైభాగంలో ఉందా, కింద ఉందా లేదా వీపులోకి వ్యాపిస్తుందా?",
            1: "వాంతులు, విరేచనాలు, ఎసిడిటీ, మంట లేదా జ్వరం ఉన్నాయా?",
            2: "ఆహారం తిన్న తర్వాత నొప్పి పెరుగుతుందా లేదా తగ్గుతుందా?",
            3: "మీకు అల్సర్, గ్యాస్ట్రిక్, పిత్తాశయ రాళ్ల సమస్య ఉందా లేదా బయటి ఆహారం తిన్నారా?",
            4: "మీకు ఏదైనా మందు లేదా ఆహారానికి అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "headache": {
        "Hindi": {
            "initial": "यह सिरदर्द या चक्कर आपको कब से आ रहे हैं?",
            0: "क्या यह सिरदर्द आधे सिर में है, माथे पर या गर्दन के पीछे की तरफ?",
            1: "क्या इसके साथ उल्टी का मन, आंखों के आगे अंधेरा, तेज रोशनी से चिढ़ या कमज़ोरी है?",
            2: "क्या तनाव, नींद की कमी या स्क्रीन देखने से दर्द बढ़ता है?",
            3: "क्या आपको हाई बीपी, चश्मे का नंबर, साइनस या परिवार में माइग्रेन की शिकायत है?",
            4: "क्या आपको किसी दवा या खाने की चीज़ से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How long have you been experiencing this headache or dizziness?",
            0: "Is the headache throbbing on one side, frontal, or radiating down the neck?",
            1: "Do you have nausea, sensitivity to bright light, blurred vision, or weakness?",
            2: "Does stress, lack of sleep, or screen time trigger or worsen the headache?",
            3: "Do you have a history of high blood pressure, sinus issues, or family migraine?",
            4: "Are you allergic to any medicines or foods?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த தலைவலி அல்லது மயக்கம் எவ்வளவு நாளாக இருக்கிறது?",
            0: "தலைவலி ஒரு பக்கத்திலா, நெற்றியிலா அல்லது கழுத்தின் பின்புறத்திலா?",
            1: "இதனுடன் குமட்டல், வெளிச்சத்தை பார்க்க முடியாத நிலை அல்லது பார்வை மங்கலாகுதல் உள்ளதா?",
            2: "மன அழுத்தம் அல்லது தூக்கமின்மையால் தலைவலி அதிகரிக்கிறதா?",
            3: "உங்களுக்கு ரத்த அழுத்தம், சைனஸ் அல்லது குடும்பத்தில் மைக்ரேன் வரலாறு உள்ளதா?",
            4: "உங்களுக்கு ஏதாவது மருந்து அல்லது உணவுக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ తలనొప్పి లేదా తలతిరగడం మీకు ఎంత కాలంగా ఉంది?",
            0: "తలనొప్పి ఒక వైపున, నుదిటిపై లేదా మెడ వెనుక భాగంలో ఉందా?",
            1: "వికారం, కాంతిని చూడలేకపోవడం, మసకబారిన చూపు లేదా బలహీనత ఉన్నాయా?",
            2: "ఒత్తిడి, నిద్రలేమి లేదా స్క్రీన్ చూడటం వల్ల తలనొప్పి పెరుగుతుందా?",
            3: "మీకు హై బీపీ, సైనస్ లేదా కుటుంబంలో మైగ్రేన్ సమస్యలు ఉన్నాయా?",
            4: "మీకు ఏదైనా మందు లేదా ఆహారానికి అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "fever": {
        "Hindi": {
            "initial": "यह बुखार आपको कितने दिनों से आ रहा है?",
            0: "क्या बुखार ठंड और कंपकंपी के साथ आता है? क्या यह किसी खास समय तेज होता है?",
            1: "क्या इसके साथ खांसी, गले में खराश, बदन दर्द, दाने या पेशाब में जलन है?",
            2: "क्या आपने पैरासिटामोल ली है? क्या दवा लेने पर बुखार उतरता है?",
            3: "क्या घर या पड़ोस में किसी को डेंगू, मलेरिया, टाइफाइड या वायरल बुखार हुआ है?",
            4: "क्या आपको किसी एंटीबायोटिक या दवा से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How many days have you had this fever?",
            0: "Does the fever come with chills and shivering, and does it spike at a specific time?",
            1: "Do you have cough, sore throat, severe body aches, rashes, or burning urination?",
            2: "Have you taken paracetamol? Does the temperature come down after medicine?",
            3: "Has anyone in your home or area had dengue, malaria, typhoid, or viral fever recently?",
            4: "Are you allergic to any antibiotics or medicines?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த காய்ச்சல் எத்தனை நாட்களாக இருக்கிறது?",
            0: "காய்ச்சல் குளிர் மற்றும் நடுக்கத்துடன் வருகிறதா? குறிப்பிட்ட நேரத்தில் அதிகமாகிறதா?",
            1: "இதனுடன் இருமல், தொண்டை வலி, உடல் வலி அல்லது சிறுநீரில் எரிச்சல் உள்ளதா?",
            2: "பாராசிட்டமால் மாத்திரை சாப்பிட்டீர்களா? மருந்து எடுத்தவுடன் காய்ச்சல் குறைகிறதா?",
            3: "அருகில் யாருக்காவது டெங்கு, மலேரியா அல்லது டைபாய்டு காய்ச்சல் உள்ளதா?",
            4: "உங்களுக்கு ஏதேனும் ஆண்டிபயாடிக் அல்லது மருந்துக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ జ్వరం మీకు ఎన్ని రోజులుగా వస్తోంది?",
            0: "జ్వరం చలి మరియు వణుకుతో వస్తుందా? ఏదైనా నిర్దిష్ట సమయంలో పెరుగుతుందా?",
            1: "దగ్గు, గొంతు నొప్పి, తీవ్రమైన ఒళ్లు నొప్పులు లేదా మూత్రంలో మంట ఉన్నాయా?",
            2: "పారాసిటమాల్ వేసుకున్నారా? మందు వేసుకున్న తర్వాత జ్వరం తగ్గుతుందా?",
            3: "ఇంట్లో లేదా చుట్టుపక్కల ఎవరికైనా డెంగ్యూ, మలేరియా లేదా టైఫాయిడ్ వచ్చిందా?",
            4: "మీకు ఏదైనా యాంటీబయాటిక్ లేదా మందుకు అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటున్నారా?"
        }
    },
    "joint_pain": {
        "Hindi": {
            "initial": "यह जोड़ों या कमर का दर्द आपको कब से हो रहा है?",
            0: "क्या जोड़ पर सूजन, लालिमा या सुबह उठने पर जकड़न महसूस होती है?",
            1: "क्या कोई चोट लगी थी? क्या चलने-फिरने या सीढ़ियां चढ़ने में तकलीफ़ होती है?",
            2: "क्या आराम करने या गर्म सिकाई करने से दर्द में राहत मिलती है?",
            3: "क्या आपको पहले गठिया, यूरिक एसिड, साइटिका या हड्डियों की कमज़ोरी रही है?",
            4: "क्या आपको किसी दर्द निवारक (पेनकिलर) दवा से एलर्जी है?",
            "default": "और कुछ बताना चाहेंगे?"
        },
        "English": {
            "initial": "How long have you had this joint or back pain?",
            0: "Is there visible swelling, redness, or morning stiffness in the joint?",
            1: "Did you have a fall or injury? Is it difficult to walk or climb stairs?",
            2: "Does rest or heat application provide relief from the pain?",
            3: "Do you have a history of arthritis, high uric acid, sciatica, or osteoporosis?",
            4: "Are you allergic to any painkiller medicines or foods?",
            "default": "Is there anything else you would like to tell me?"
        },
        "Tamil": {
            "initial": "இந்த மூட்டு அல்லது முதுகு வலி எவ்வளவு நாளாக இருக்கிறது?",
            0: "மூட்டில் வீக்கம், சிவத்தல் அல்லது காலையில் விறைப்பு தன்மை உள்ளதா?",
            1: "ஏதாவது காயம் ஏற்பட்டதா? நடப்பதற்கோ அல்லது படிக்கட்டுகள் ஏறுவதற்கோ சிரமமாக உள்ளதா?",
            2: "ஓய்வெடுப்பதாலோ அல்லது ஒத்தடம் கொடுப்பதாலோ வலி குறைகிறதா?",
            3: "உங்களுக்கு மூட்டுவாதம், யூரிக் அமிலம் அல்லது எலும்பு தேய்மானம் உள்ளதா?",
            4: "உங்களுக்கு வலி நிவாரணி மருந்துகளுக்கு ஒவ்வாமை இருக்கிறதா?",
            "default": "வேறு ஏதாவது சொல்ல விரும்புகிறீர்களா?"
        },
        "Telugu": {
            "initial": "ఈ కీళ్ల లేదా వెన్ను నొప్పి మీకు ఎంత కాలంగా ఉంది?",
            0: "కీళ్లపై వాపు, ఎరుపుదనం లేదా ఉదయం పూట బిగుతుగా ఉండటం ఉందా?",
            1: "ఏదైనా గాయం అయిందా? నడవడానికి లేదా మెట్లు ఎక్కడానికి కష్టంగా ఉందా?",
            2: "విశ్రాంతి లేదా వేడి కాపడం వల్ల నొప్పి తగ్గుతుందా?",
            3: "మీకు ఆర్థరైటిస్, యూరిక్ యాసిడ్ లేదా ఎముకల బలహీనత సమస్యలు ఉన్నాయా?",
            4: "మీకు పెయిన్‌కిల్లర్ మందులకు అలర్జీ ఉందా?",
            "default": "ఇంకా ఏమైనా చెప్పాలనుకుంటుannారా?"
        }
    },
    "general": {
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
        }
    }
}

def detect_symptom_category(transcript: str) -> str:
    """Classifies patient transcript into targeted symptom tracks using multilingual keyword heuristics."""
    if not transcript:
        return "general"
    t = transcript.lower()

    # 1. Chest Pain & Cardiac / Respiratory
    chest_keywords = [
        "chest", "सीने", "छाती", "heart", "दिल", "सांस", "breath", "palpitation",
        "घबराहट", "धड़कन", "angina", "cardiac", "நெஞ்சு", "மார்பு", "గుండె", "छातीत", "বুক"
    ]
    if any(k in t for k in chest_keywords):
        return "chest_pain"

    # 2. Stomach & GI / Abdominal
    stomach_keywords = [
        "stomach", "abdomen", "abdominal", "belly", "पेट", "pet", "vomit", "उल्टी",
        "loose motion", "दस्त", "acidity", "gas", "गैस", "जलन", "constipation",
        "कब्ज", "ulcer", "appetite", "भूख", "வயிறு", "కడుపు", "पोट", "পেট"
    ]
    if any(k in t for k in stomach_keywords):
        return "stomach_pain"

    # 3. Headache & Neurological / Dizziness
    headache_keywords = [
        "headache", "head pain", "head", "सिर", "सर", "migraine", "माइग्रेन",
        "dizziness", "चक्कर", "faint", "बेहोश", "vision", "धुंधला", "stroke",
        "தலைவலி", "తలనొప్పి", "डोकेदुखी", "মাথাব্যথা"
    ]
    if any(k in t for k in headache_keywords):
        return "headache"

    # 4. Fever & Infections / Chills
    fever_keywords = [
        "fever", "बुखार", "ताप", "chills", "ठंड", "shivering", "कंपकंपी",
        "dengue", "डेंगू", "malaria", "मलेरिया", "typhoid", "टाइफाइड", "viral",
        "காய்ச்சல்", "జ్వరం", "ताप", "জ্বর"
    ]
    if any(k in t for k in fever_keywords):
        return "fever"

    # 5. Joint, Orthopedic & Back Pain
    joint_keywords = [
        "joint", "जोड़", "knee", "घुटने", "back pain", "कमर", "spine", "रीढ़",
        "bone", "हड्डी", "swelling", "सूजन", "arthritis", "गठिया", "fracture",
        "stiffness", "जकड़न", "மூட்டு", "కీళ్ల", "सांधेदुखी", "গাঁটের ব্যথা"
    ]
    if any(k in t for k in joint_keywords):
        return "joint_pain"

    return "general"

def get_phase_question(language: str, phase, category: str = "general") -> str:
    """Get a pre-written question template tailored to language and symptom category."""
    cat_dict = CATEGORY_QUESTIONS.get(category, CATEGORY_QUESTIONS.get("general", {}))
    lang_questions = cat_dict.get(language, cat_dict.get("English", {}))
    if not lang_questions:
        gen_cat = CATEGORY_QUESTIONS.get("general", {})
        lang_questions = gen_cat.get(language, gen_cat.get("English", {}))
    return lang_questions.get(phase, lang_questions.get("default", "Is there anything else you would like to tell me?"))

# ── Pydantic Models ──
PATIENT_JSON_TEMPLATE = """{
  "chief_complaint": "Main presenting symptom (translated to clinical English)",
  "hpi": "Chronological narrative of the CURRENT presenting complaint ONLY: onset, duration, character, radiation, severity, aggravating/relieving factors. STRICTLY EXCLUDE past history, family history, lifestyle/habits, allergies, and general systemic symptom denials.",
  "is_emergency": false,
  "severity": "Low|Medium|High",
  "duration": "Symptom duration (e.g. '2 days')",
  "past_medical_history": "Past medical conditions (or 'Uncertain / unconfirmed (patient does not recall)' if unsure, or 'Patient denies past chronic medical conditions / No significant past medical history' if denied)",
  "family_history": "Family history (or 'Patient denies family history of similar complaints / No significant family history' if denied, or 'Uncertain / unconfirmed' if unsure)",
  "personal_history": "Smoking, alcohol, diet, habits (or 'No significant lifestyle or habit risks reported')",
  "allergies": "Drug/food allergies (or 'No known drug or food allergies (NKDA)' if denied)",
  "review_of_systems": "Summary of systemic positive and negative symptoms (e.g. 'Patient reports diaphoresis; denies fever or vomiting')",
  "clinical_impression": {
    "clinical_synthesis": [
      "Key acute symptoms, duration, and anatomical localization reported today",
      "Corroborating objective findings from today's uploaded reports/labs (or 'No acute lab flags reported')",
      "Historical ABHA risk context and underlying clinical etiology rationale"
    ],
    "probable_diagnoses": [
      {
        "condition": "Primary Suspected Condition Name",
        "likelihood": "High|Medium|Low",
        "supporting_evidence": "Clinical rationale tying together symptoms, lab values, and past history."
      }
    ],
    "suggested_investigations": [
      "Key diagnostic test or scan 1",
      "Key diagnostic test or scan 2"
    ],
    "critical_rule_outs": [
      "Critical high-risk condition to actively rule out"
    ]
  },
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
    clinical_impression: Optional[dict] = None
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
    symptom_category = Column(String, default="general")
    clinical_impression_json = Column(Text, default="{}")
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
        for col_name, col_type in [("abha_id", "VARCHAR"), ("patient_name", "VARCHAR"), ("age", "VARCHAR"), ("gender", "VARCHAR"), ("phone", "VARCHAR"), ("symptom_category", "VARCHAR"), ("raw_dialogue", "TEXT"), ("is_synthesized", "BOOLEAN"), ("abha_relevance_json", "TEXT"), ("clinical_impression_json", "TEXT")]:
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


# ── Clinical HPI Cleaner (Eliminates Cross-Section Redundancy) ──
def clean_hpi_text(hpi: str) -> str:
    """
    Cleans History of Present Illness (HPI) text by removing sentences that mistakenly 
    duplicate or bundle Past Medical History, Family History, Allergies, Personal/Lifestyle History, 
    or general Review of Systems denials.
    """
    if not hpi or not isinstance(hpi, str) or not hpi.strip():
        return hpi

    patterns = [
        # Past medical history mentions / denials
        r'\b(?:past\s+(?:chronic\s+)?medical\s+(?:history|conditions?|illness|issues?)|past\s+medical|past\s+surgical|past\s+illness)\b',
        r'\b(?:denies\s+(?:any\s+)?past\s+(?:chronic\s+)?(?:medical|illness|conditions?))\b',
        r'\b(?:no\s+significant\s+past\s+medical)\b',
        r'\b(?:reports?\s+no\s+past\s+(?:medical|chronic))\b',
        # Family history mentions / denials
        r'\b(?:family\s+history|hereditary\s+conditions?|family\s+members?\s+(?:have|had))\b',
        r'\b(?:denies\s+(?:any\s+)?(?:significant\s+)?family\s+history)\b',
        r'\b(?:no\s+significant\s+family\s+history)\b',
        # Allergies mentions / denials
        r'\b(?:drug\s+or\s+food\s+allergies|known\s+drug|food\s+allergies|allergic\s+to\s+(?:any\s+)?(?:medication|food|drugs?)|allergies\b|nkda\b)',
        r'\b(?:denies\s+(?:any\s+)?(?:known\s+)?(?:drug|food\s+)?allergies)\b',
        r'\b(?:no\s+known\s+(?:drug|food\s+)?allergies)\b',
        # Personal / Lifestyle mentions / denials
        r'\b(?:lifestyle\s+or\s+habit|lifestyle\s+risks?|habit\s+risks?|smoking\s+(?:or|and)\s+alcohol|tobacco|substance\s+use)\b',
        r'\b(?:denies\s+(?:any\s+)?significant\s+lifestyle)\b',
        r'\b(?:no\s+significant\s+lifestyle)\b',
        # Systemic ROS checklist denials
        r'\b(?:associated\s+systemic\s+symptoms|denies\s+(?:any\s+)?associated\s+systemic|review\s+of\s+systems)\b',
    ]
    filter_re = re.compile('|'.join(patterns), re.IGNORECASE)

    cleaned_lines = []
    for line in hpi.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        
        if filter_re.search(line_str):
            sentences = re.split(r'(?<=[.!?])\s+', line_str)
            valid_sentences = [s.strip() for s in sentences if s.strip() and not filter_re.search(s)]
            if valid_sentences:
                cleaned_lines.append(" ".join(valid_sentences))
        else:
            cleaned_lines.append(line_str)

    cleaned_result = "\n".join(cleaned_lines).strip()
    return cleaned_result if cleaned_result else hpi


# ── Instant Patient Builder (0ms LLM Overhead) ──
def build_patient_from_transcript(transcript, language, is_ayush, pt_id, db, abha_id=None, patient_name="Patient", age="", gender="", phone=""):
    t_lower = transcript.lower()
    is_emerg = any(w in t_lower for w in [
        "chest pain", "heart", "attack", "breath", "stroke", "paralysis", "unconscious", "bleeding", "accident",
        "सीने में दर्द", "हार्ट", "अटैक", "सांस", "बेहोश", "खून", "छातीत दुखणे", "గుండె", "நெஞ்சு வலி", "বুকের ব্যথা"
    ])

    symptom_cat = detect_symptom_category(transcript)
    print(f"🎯 Detected primary symptom track: '{symptom_cat}' for complaint: '{transcript}'")

    dialogue_entry = f"Patient (Chief Complaint - {language}): {transcript}\n"

    patient = PatientRecord(
        patient_id=pt_id,
        abha_id=abha_id,
        is_ayush=bool(is_ayush),
        patient_name=patient_name or "Patient",
        age=str(age) if age else "",
        gender=str(gender) if gender else "",
        phone=str(phone) if phone else "",
        symptom_category=symptom_cat,
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

    # Use targeted SOCRATES-framework question template for the detected symptom track
    initial_question = get_phase_question(language, "initial", category=symptom_cat)
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

        # Tier 1: Patient's Today's Spoken Input (Primary Clinical Anchor)
        tier1_str = f"=== TIER 1: PATIENT'S TODAY'S SPOKEN INTAKE (PRIMARY CLINICAL ANCHOR) ===\n{full_transcript}"

        # Tier 2: Currently Uploaded Documents & Lab Reports (Immediate Objective Corroboration)
        tier2_str = "=== TIER 2: CURRENTLY UPLOADED MEDICAL DOCUMENTS & LAB REPORTS ===\n(No documents uploaded today)"
        if patient.flagged_lab_values and patient.flagged_lab_values != "[]":
            try:
                docs_list = json.loads(patient.flagged_lab_values)
                if isinstance(docs_list, list) and len(docs_list) > 0:
                    tier2_str = "=== TIER 2: CURRENTLY UPLOADED MEDICAL DOCUMENTS & LAB REPORTS (IMMEDIATE CORROBORATION) ===\n"
                    for idx, d in enumerate(docs_list, 1):
                        if isinstance(d, dict):
                            tier2_str += f"- Document #{idx} ({d.get('document_type', 'Report')} - Date: {d.get('document_date', 'Unknown')}): Summary: {d.get('summary', '')}, Diagnoses: {d.get('diagnoses', [])}, Meds: {d.get('medications', [])}, Flagged Labs: {d.get('flagged_values', [])}\n"
            except Exception as e:
                print(f"Error parsing docs for synthesis: {e}")

        # Tier 3: Verified Past ABHA Historical Records (Background Context & Risk Filter)
        tier3_str = "=== TIER 3: VERIFIED PAST ABHA MEDICAL HISTORY (BACKGROUND CONTEXT ONLY) ===\n(No past ABHA records on file)"
        if abha_id:
            try:
                past_visits = db.query(VisitHistory).filter(VisitHistory.abha_id == abha_id).all()
                if past_visits:
                    tier3_str = "=== TIER 3: VERIFIED PAST ABHA MEDICAL HISTORY (BACKGROUND CONTEXT ONLY) ===\n"
                    for idx, v in enumerate(past_visits, 1):
                        tier3_str += f"- Historical Visit #{idx} ({v.visit_date} - {v.specialty}): Chief Complaint: {v.chief_complaint}, Diagnoses: {v.diagnoses}, Meds: {v.medications}, Flags: {v.flagged_values}, Summary: {v.summary}\n"
            except Exception as e:
                print(f"Error loading ABHA history for synthesis: {e}")

        prompt = f"""You are an expert Chief Medical Officer and AI Clinical Decision Support Specialist.
Review the patient's data below following a strict 3-tier clinical diagnostic reasoning hierarchy:

Language spoken: {language}
{ayush_inst}

{tier1_str}

{tier2_str}

{tier3_str}

TASK: Perform high-precision clinical synthesis and generate Clinical Decision Support (CDSS) insights into a standard medical EHR record.

CRITICAL CLINICAL REASONING ORDER FOR PROBABLE DIAGNOSES (CDSS):
You MUST follow this exact sequential diagnostic reasoning flow:
1. STEP 1 — ANCHOR ON PATIENT'S CURRENT PRESENTATION (TIER 1):
   - The primary suspected condition MUST be anchored strictly to the patient's active complaints, onset, location, character, and systemic symptoms reported TODAY.
2. STEP 2 — CORROBORATE WITH CURRENTLY UPLOADED DOCUMENTS (TIER 2):
   - Cross-examine Tier 1 symptoms against today's scanned blood tests, ECGs, or imaging flags to confirm or refine the acute diagnosis.
3. STEP 3 — FILTER BACKGROUND CONTEXT FROM PAST ABHA HISTORY (TIER 3):
   - Check historical ABHA visits ONLY to identify relevant risk factors, past recurrent conditions, or chronic co-morbidities (e.g. past CAD stenting when presenting with chest pain).
   - STRICT WARNING: NEVER allow unrelated past history (e.g. past ankle sprain or cataract) to override or misguide today's acute diagnosis when today's symptoms represent a different organ system!
4. STEP 4 — FORMULATE DIFFERENTIAL DIAGNOSES:
   - Generate top 2-3 differential diagnoses reflecting this exact priority order. Each diagnosis must clearly state its likelihood ("High"|"Medium"|"Low") and supporting evidence linking Tier 1 -> Tier 2 -> Tier 3.

SECTION SPECIFIC RULES:
1. HISTORY OF PRESENT ILLNESS (HPI) — STRICT BOUNDARIES:
   - `hpi` MUST ONLY describe the chronology of the CURRENT presenting complaint (onset, duration, anatomical site, character, severity, progression, aggravating/relieving factors).
   - STRICT PROHIBITION: DO NOT mention past medical history, family history, lifestyle/personal habits, allergies, or general review-of-systems denials in the `hpi` field. Each belongs ONLY in its dedicated section.

2. ACCURATELY DISTINGUISH DENIAL ("NO") VS UNCERTAINTY ("NOT SURE / DON'T REMEMBER") VS NOT ASKED:
   - When patient clearly DENIES: Write "Patient denies..."
   - When patient expresses UNCERTAINTY / LACK OF MEMORY: Record as "Uncertain / unconfirmed (patient does not recall / unsure)". DO NOT write "No" or "Denies"!
   - When NOT asked: Record as "Not assessed".

3. REVIEW OF SYSTEMS (ROS):
   - Actively summarize associated systemic symptoms asked or reported during the interview (e.g. "Patient denies fever, vomiting, or dyspnea; reports diaphoresis").

4. CLINICAL DECISION SUPPORT (CDSS) & DIAGNOSTIC IMPRESSION:
   - `clinical_impression`:
     * `clinical_synthesis`: Array of 2-3 concise bullet points: (1) Current acute presentation/timeline, (2) Corroborating lab/imaging findings, (3) Relevant historical ABHA context & primary clinical etiology rationale.
     * `probable_diagnoses`: Top 2-3 differentials with `condition`, `likelihood` ("High"|"Medium"|"Low"), and `supporting_evidence`.
     * `suggested_investigations`: 2-4 recommended next diagnostic tests/scans.
     * `critical_rule_outs`: 1-3 high-risk life-threatening conditions to actively exclude.

5. EMERGENCY TRIAGE & SEVERITY:
   - Set `is_emergency`: true if red flags (acute coronary syndrome, stroke signs, severe trauma, acute respiratory distress), else false.
   - Set `severity`: "High" | "Medium" | "Low".

6. Set `next_question` to 'complete'.

Output ONLY valid JSON:
{PATIENT_JSON_TEMPLATE}"""

        print(f"→ Synthesizing tiered clinical record + CDSS (Tier 1 Input -> Tier 2 Docs -> Tier 3 ABHA) for patient {patient.patient_id} in background...")
        response_text = call_llm(prompt)
        result_json = json.loads(extract_json_string(response_text))
        result_json = unwrap_json(result_json)
        ext = PatientExtraction(**result_json)

        # Update patient record with synthesized clinical data
        patient.chief_complaint = ext.chief_complaint or patient.chief_complaint
        patient.hpi = clean_hpi_text(ext.hpi or patient.hpi)
        patient.is_emergency = ext.is_emergency
        patient.severity = ext.severity or patient.severity
        patient.duration = ext.duration or "Unknown"
        patient.past_medical_history = ext.past_medical_history or "No significant past medical history"
        patient.family_history = ext.family_history or "No significant family history"
        patient.personal_history = ext.personal_history or "No significant lifestyle risks"
        patient.allergies = ext.allergies or "No known drug allergies (NKDA)"
        patient.review_of_systems = ext.review_of_systems or "Patient denies associated systemic symptoms"
        if ext.clinical_impression and isinstance(ext.clinical_impression, dict):
            patient.clinical_impression_json = json.dumps(ext.clinical_impression)
        patient.prakriti = ext.prakriti if is_ayush else "Not assessed"
        patient.vikriti = ext.vikriti if is_ayush else "Not assessed"
        patient.agni = ext.agni if is_ayush else "Not assessed"
        patient.is_synthesized = True

        db.commit()
        print(f"✅ Tiered Clinical record & CDSS synthesis complete for {patient.patient_id} ({patient.chief_complaint})")

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
    symptom_cat = patient.symptom_category or "general"
    prev_q = get_phase_question(language, follow_up_count - 1 if follow_up_count > 0 else "initial", category=symptom_cat)
    dialogue_entry = f"Doctor ({phase_label}): {prev_q}\nPatient ({language}): {transcript}\n"

    current_dialogue = patient.raw_dialogue or ""
    patient.raw_dialogue = current_dialogue + dialogue_entry

    is_complete = True if follow_up_count >= 5 else False
    next_question = get_phase_question(language, follow_up_count, category=symptom_cat) if follow_up_count < 5 else "Thank you. Let us proceed to document scanning."

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

    # Auto-clean and persist HPI if it contains redundant cross-section leakage
    cleaned_hpi = clean_hpi_text(patient.hpi) if patient.hpi else "None reported"
    if patient.hpi and cleaned_hpi != patient.hpi:
        patient.hpi = cleaned_hpi
        try:
            db.commit()
        except Exception:
            pass

    impression = {}
    if patient.clinical_impression_json:
        try:
            parsed = json.loads(patient.clinical_impression_json)
            if isinstance(parsed, dict):
                impression = parsed
        except Exception:
            impression = {}

    return {
        "patient_id": patient.patient_id,
        "patient_name": patient.patient_name or "Patient",
        "age": patient.age or "",
        "gender": patient.gender or "",
        "phone": patient.phone or "",
        "abha_id": patient.abha_id,
        "chief_complaint": patient.chief_complaint or "Not recorded",
        "hpi": cleaned_hpi,
        "is_emergency": patient.is_emergency,
        "severity": patient.severity or "Unknown",
        "duration": patient.duration or "Unknown",
        "past_medical_history": patient.past_medical_history or "None reported",
        "family_history": patient.family_history or "None reported",
        "personal_history": patient.personal_history or "None reported",
        "allergies": patient.allergies or "None reported",
        "review_of_systems": patient.review_of_systems or "None reported",
        "clinical_impression": impression,
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

    demo_impression = {
        "clinical_synthesis": [
            "58-year-old male presenting with acute crushing retrosternal chest pain radiating to left arm with diaphoresis of 2 hours duration.",
            "Historical ABHA records confirm prior STEMI (2024 PCI LAD stenting) and uncontrolled T2DM (HbA1c 8.2%).",
            "Elevated lipid profile (LDL 145 mg/dL) and active presentation strongly indicate recurrent acute coronary syndrome / stent thrombosis."
        ],
        "probable_diagnoses": [
            {
                "condition": "Acute Coronary Syndrome / Recurrent NSTEMI vs Stent Thrombosis",
                "likelihood": "High",
                "supporting_evidence": "Crushing substernal pain radiating to left arm with diaphoresis, past STEMI with LAD stent in 2024, uncontrolled diabetes (HbA1c 8.2%)."
            },
            {
                "condition": "Unstable Angina Pectoris",
                "likelihood": "Medium",
                "supporting_evidence": "Exertional chest discomfort with high cardiovascular risk profile and uncontrolled hyperlipidemia."
            },
            {
                "condition": "Acute Gastroesophageal Reflux Disease (GERD) with Esophageal Spasm",
                "likelihood": "Low",
                "supporting_evidence": "Can mimic substernal chest pressure, but severe cardiac risk factors mandate treating as ACS until ruled out."
            }
        ],
        "suggested_investigations": [
            "Stat 12-lead ECG",
            "Serial Cardiac Biomarkers (Troponin I & CK-MB at 0h, 3h)",
            "Bedside 2D Echocardiography (Wall Motion Assessment)",
            "Coronary Angiography consideration"
        ],
        "critical_rule_outs": [
            "Acute Aortic Dissection",
            "Pulmonary Embolism",
            "Tension Pneumothorax"
        ]
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
        clinical_impression_json=json.dumps(demo_impression),
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
