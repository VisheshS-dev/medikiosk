import os
import hashlib
from gtts import gTTS
from main import CATEGORY_QUESTIONS, LANGUAGE_CODES

TTS_CACHE_DIR = os.path.join("uploads", "tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)

# Standard greetings & UI prompt phrases
common_phrases = [
    # Hindi Greetings & System Prompts
    ("नमस्ते! मैं MediKiosk AI हूँ। मैं आपकी चिकित्सा जानकारी एकत्र करने में मदद करूँगा।\n\nआपको आज डॉक्टर से मिलने की क्या समस्या है?", "hi"),
    ("नमस्ते! मैं MediKiosk AI हूँ। मैं आपकी चिकित्सा जानकारी एकत्र करने में मदद करूँगा। आपको आज डॉक्टर से मिलने की क्या समस्या है?", "hi"),
    ("धन्यवाद। आइए अब आपके दस्तावेज़ या रिपोर्ट स्कैन करते हैं।", "hi"),
    ("आपकी जानकारी सफलतापूर्वक दर्ज कर ली गई है। डॉक्टर से मिलने के लिए पर्ची लें।", "hi"),

    # English Greetings & System Prompts
    ("Hello! I am MediKiosk AI. I will assist you in collecting your preliminary medical information.\n\nWhat brings you to see the doctor today?", "en"),
    ("Hello! I am MediKiosk AI. I will assist you in collecting your preliminary medical information. What brings you to see the doctor today?", "en"),
    ("Thank you. Let us proceed to document scanning.", "en"),
    ("Thank you for providing all the information.", "en"),
    ("Your consultation details have been recorded. Please collect your token for the doctor.", "en"),
]

phrases_to_generate = set(common_phrases)

# Extract all symptom-specific questions from CATEGORY_QUESTIONS
for category_name, lang_dict in CATEGORY_QUESTIONS.items():
    for lang_name, questions in lang_dict.items():
        lang_code = LANGUAGE_CODES.get(lang_name, "en")
        for key, q_text in questions.items():
            if isinstance(q_text, str) and q_text.strip():
                phrases_to_generate.add((q_text.strip(), lang_code))

print(f"Total unique phrases to cache across all categories: {len(phrases_to_generate)}")

success_count = 0
exists_count = 0
error_count = 0

for idx, (text, lang) in enumerate(sorted(phrases_to_generate, key=lambda x: (x[1], x[0])), 1):
    cache_key = hashlib.md5(f"{lang}_{text.strip()}".encode("utf-8")).hexdigest()
    out_file = os.path.join(TTS_CACHE_DIR, f"{cache_key}.mp3")
    if not os.path.exists(out_file):
        try:
            print(f"[{idx}/{len(phrases_to_generate)}] Generating [{lang}] '{text[:45]}...'")
            tts = gTTS(text=text.strip(), lang=lang)
            tts.save(out_file)
            success_count += 1
        except Exception as e:
            print(f"❌ Error generating phrase [{lang}] '{text[:30]}': {e}")
            error_count += 1
    else:
        exists_count += 1

print(f"\n🎉 Pre-generation complete! Generated: {success_count}, Already Cached: {exists_count}, Errors: {error_count}")

