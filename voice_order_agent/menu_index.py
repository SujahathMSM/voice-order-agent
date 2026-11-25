"""Step 2b: embed the menu once and save the vectors to disk."""
import json
from pathlib import Path

import numpy as np
from mistralai.client import Mistral

from .config import Settings, load_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MENU_JSON = PROJECT_ROOT / "data" / "menu.json"
EMBED_MODEL = "mistral-embed"


def build_menu_text(item: dict) -> str:
    """One text per item: everything a caller might say about it."""
    parts = [item["name"]]
    parts += item.get("aliases", [])
    parts += item.get("options", [])
    parts.append(item["description"])
    return ". ".join(parts)


def embed_menu(settings: Settings, menu_items: list[dict]) -> None:
    """Call the embeddings API ONCE with all texts, save vectors + ids."""
    client = Mistral(api_key=settings.mistral_api_key)

    texts = [build_menu_text(item) for item in menu_items]
    response = client.embeddings.create(model=EMBED_MODEL, inputs=texts)

    # Order is guaranteed: response.data[i] matches texts[i]
    vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
    ids = [item["id"] for item in menu_items]

    # Normalize: after this, "similarity" is a simple dot product later.
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    settings.menu_vector_db.parent.mkdir(parents=True, exist_ok=True)
    np.savez(settings.menu_vector_db, vectors=vectors, ids=np.array(ids))
    print(f"Saved {len(ids)} item vectors to {settings.menu_vector_db}")


def load_vectors(settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Load what embed_menu saved. Used by search_menu in step 2c."""
    data = np.load(settings.menu_vector_db)
    return data["vectors"], data["ids"]


def main() -> None:
    """Run once: python -m voice_order_agent.menu_index"""
    settings = load_settings()
    menu = json.loads(MENU_JSON.read_text(encoding="utf-8"))["items"]
    embed_menu(settings, menu)


if __name__ == "__main__":
    main()
