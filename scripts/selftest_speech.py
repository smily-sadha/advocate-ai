"""
selftest_speech.py — quick self-test for the speech stack.
Run from the project root with the venv python.

1. Imports the FastAPI app (verifies all routes/services load)
2. gTTS: synthesizes a sentence to mp3
3. Whisper large-v3: transcribes that mp3 back to text (round-trip)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
os.chdir(BACKEND)
sys.path.insert(0, BACKEND)

print("[1] Importing app...")
import main  # noqa: E402
print("    OK ->", main.app.title)

from config import settings  # noqa: E402
from services.tts import synthesize_speech  # noqa: E402
from services.stt import transcribe_audio  # noqa: E402

SENTENCE = "Section 80C of the Income Tax Act allows deductions on certain investments."

print(f"[2] gTTS synth (engine={settings.tts_engine}, lang={settings.tts_lang})...")
out = synthesize_speech(
    text=SENTENCE,
    output_dir=settings.audio_output_dir,
    engine=settings.tts_engine,
    lang=settings.tts_lang,
)
size = os.path.getsize(out)
print(f"    OK -> {out} ({size} bytes)")
assert out.endswith(".mp3") and size > 0, "gTTS produced no/empty mp3"

print(f"[3] Whisper transcribe (model={settings.whisper_model}) — may download ~3GB...")
res = transcribe_audio(audio_path=out, model_size=settings.whisper_model)
print(f"    detected lang: {res['language']}")
print(f"    transcript   : {res['text']}")

print("\nALL SPEECH SELF-TESTS PASSED.")
