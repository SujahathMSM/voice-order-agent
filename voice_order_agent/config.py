"""Configuration: the only module that reads .env / environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of this package), regardless of
# where the script is launched from.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Settings:
    """All configuration the project needs. Add fields here, never os.environ elsewhere."""
    # Mistral API (Voxtral STT/TTS, Mistral Small, embeddings)
    mistral_api_key: str

    # LLM proxy (step 9): OpenAI-compatible endpoint for Mistral Small / Meta Muse
    proxy_base_url: str
    fast_model: str = "mistral-small-latest"
    strong_model: str = "muse"  # exact model name TBD in step 9

    # Vector store file for the menu (filled in step 2)
    menu_vector_db: Path = Path(__file__).resolve().parent.parent / "data" / "menu_vectors.npz"


def load_settings() -> Settings:
    """Read required secrets from the environment, fail loudly if missing."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Check your .env file at the project root."
        )
    return Settings(
        mistral_api_key=api_key,
        proxy_base_url=os.environ.get("LLM_PROXY_BASE_URL", ""),
    )
