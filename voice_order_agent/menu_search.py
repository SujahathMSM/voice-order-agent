"""Step 2c: search the menu two ways - fuzzy text match + vector meaning match."""
import json
from pathlib import Path

import numpy as np
from mistralai.client import Mistral
from rapidfuzz import process, fuzz

from .config import Settings, load_settings
from .menu_index import EMBED_MODEL, load_vectors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MENU_JSON = PROJECT_ROOT / "data" / "menu.json"


def load_menu() -> dict[str, dict]:
    """Menu items by id: {'kottu-chicken': {...}, ...}"""
    items = json.loads(MENU_JSON.read_text(encoding="utf-8"))["items"]
    return {item["id"]: item for item in items}


class MenuSearch:
    def __init__(self, settings: Settings) -> None:
        self.items = load_menu()
        self.client = Mistral(api_key=settings.mistral_api_key)
        self.vectors, self.ids = load_vectors(settings)

        # Fuzzy vocabulary: every name + alias -> the item id it belongs to.
        # Example: "coke" -> "soda", "kiribath" -> "rice-milk"
        self.fuzzy_names: dict[str, str] = {}
        for item in self.items.values():
            for text in [item["name"]] + item.get("aliases", []):
                self.fuzzy_names[text.lower()] = item["id"]

    def _vector_scores(self, query: str) -> dict[str, float]:
        """Embed the caller's text (1 API call), compare with all menu vectors."""
        response = self.client.embeddings.create(model=EMBED_MODEL, inputs=[query])
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        # Vectors were normalized in step 2b, so a dot product IS the similarity.
        scores = self.vectors @ query_vec                   # 30 numbers
        return dict(zip(self.ids.tolist(), scores.tolist()))

    def _fuzzy_scores(self, query: str, limit: int = 3) -> dict[str, float]:
        """Compare text against names+aliases. Free (no API), scores 0-100."""
        hits = process.extract(
            query.lower(), self.fuzzy_names.keys(),
            scorer=fuzz.partial_ratio, limit=limit,
        )
        # hits = [(matched_text, score, ...)], map text -> id -> get best per id
        best: dict[str, float] = {}
        for text, score, _ in hits:
            item_id = self.fuzzy_names[text]
            best[item_id] = max(best.get(item_id, 0.0), score / 100.0)
        return best

    def search(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        """Return the top-k menu items with a combined 0-1 score."""
        combined: dict[str, float] = self._fuzzy_scores(query)
        for item_id, vscore in self._vector_scores(query).items():
            combined[item_id] = max(combined.get(item_id, 0.0), vscore)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.items[item_id], score) for item_id, score in ranked]


def main() -> None:
    """Quick test: python -m voice_order_agent.menu_search"""
    settings = load_settings()
    searcher = MenuSearch(settings)
    for query in [
        "coke",                                    # fuzzy should nail this
        "chicken second code instead of three",    # the kottu mangle from step 0
        "something without onions",                # meaning, no name match
        "how much is the seafloor flight less",    # whisper's mangle of q03
    ]:
        print(f"\nQ: {query}")
        for item, score in searcher.search(query):
            print(f"   {score:4.2f}  {item['name']}")


if __name__ == "__main__":
    main()
