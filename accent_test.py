"""Step 0: compare Voxtral vs faster-whisper on my own recordings."""
import json
from pathlib import Path

from dotenv import load_dotenv
import jiwer
from faster_whisper import WhisperModel
from mistralai.client import Mistral
import os

load_dotenv()                      # reads MISTRAL_API_KEY from .env

AUDIO_DIR = Path("audio")
REFS = json.loads(Path("refs.json").read_text(encoding="utf-8"))


def transcribe_voxtral(client: Mistral, wav: Path) -> str:
    """One API call per file. Mistral docs: audio.transcriptions.complete."""
    with wav.open("rb") as f:
        res = client.audio.transcriptions.complete(
            model="voxtral-mini-latest",
            file={"content": f, "file_name": wav.name},
            language="en",
        )
    return res.text.strip()


def transcribe_whisper(model: WhisperModel, wav: Path) -> str:
    """Runs fully on CPU the first time; downloads ~500MB model on first use."""
    segments, _info = model.transcribe(str(wav), language="en")
    return " ".join(seg.text for seg in segments).strip()


def wer(reference: str, hypothesis: str) -> float:
    """jiwer handles alignment; we lowercase so 'What' == 'what'."""
    return jiwer.wer(reference.lower(), hypothesis.lower())


def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    # "small" matches your plan; int8 = fastest CPU mode
    whisper = WhisperModel("small", device="cpu", compute_type="int8")

    rows = []
    for wav_name, ref in REFS.items():
        wav = AUDIO_DIR / wav_name
        vox_text = transcribe_voxtral(client, wav)
        whisper_text = transcribe_whisper(whisper, wav)
        rows.append((wav_name, wer(ref, vox_text), wer(ref, whisper_text),
                      vox_text, whisper_text))

    print(f"{'file':<10}{'voxtral WER':<14}{'whisper WER':<14}")
    for name, v_wer, w_wer, _, _ in rows:
        print(f"{name:<10}{v_wer:<14.2%}{w_wer:<14.2%}")
    print(f"\nAVERAGE  voxtral={sum(r[1] for r in rows)/len(rows):.2%}"
          f"  whisper={sum(r[2] for r in rows)/len(rows):.2%}\n")
    for name, _, _, vox_text, whisper_text in rows:
        print(f"--- {name} ---\nvoxtral: {vox_text}\nwhisper: {whisper_text}\n")


if __name__ == "__main__":
    main()