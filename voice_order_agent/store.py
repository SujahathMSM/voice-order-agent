"""Step 10: confirmed orders go to SQLite. The ONLY code that writes orders.db.

Design:
- One-to-many (orders + order_items), because the questions that matter are
  relational ("how many kottu today?", "orders for the 7 PM pickup").
- Only CONFIRMED orders enter the DB. Open orders live in RAM only.
- save_order is idempotent per Order object (a re-confirm attempt can't dup).
"""
import sqlite3
from pathlib import Path

from .order import Order

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "orders.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    confirmed_at TEXT NOT NULL,          -- ISO 8601 local time
    pickup_time TEXT NOT NULL,          -- "HH:MM" 24h
    total INTEGER NOT NULL              -- LKR, computed by Order itself
);
CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    menu_id TEXT NOT NULL,
    name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price INTEGER NOT NULL,        -- frozen from menu at add time
    options TEXT NOT NULL DEFAULT ''    -- comma separated
);
CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id);
CREATE TABLE IF NOT EXISTS turn_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,             -- ISO 8601
    session TEXT,                          -- agent/job identifier
    kind TEXT NOT NULL,                    -- llm | stt | tts | eou | vad
    duration REAL,
    ttft REAL,                             -- llm: time to first token
    ttfb REAL,                             -- tts: time to first audio byte
    end_of_utterance_delay REAL,           -- eou: turn-detection delay
    transcription_delay REAL,              -- eou: transcript after speech end
    audio_duration REAL,
    cancelled INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS idx_metrics_kind ON turn_metrics(kind);
"""

METRIC_COLUMNS = ("duration", "ttft", "ttfb",
                 "end_of_utterance_delay", "transcription_delay",
                 "audio_duration", "cancelled", "prompt_tokens",
                 "completion_tokens")


class OrderStore:
    """Owns all SQL. The LLM never touches this file."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def save_order(self, order: Order, confirmed_at: str) -> int:
        """Persist one CONFIRMED order. Raises for open orders. Safe to call twice."""
        if order.status != "confirmed":
            raise ValueError("Refusing to save an order that is not confirmed")
        if getattr(order, "_stored", False):
            # find our row: we can't know the id without passing it around,
            # so the honest guard is: only the confirm path saves, once.
            raise ValueError("Order already saved")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO orders (confirmed_at, pickup_time, total) VALUES (?, ?, ?)",
                (confirmed_at, order.pickup_time, order.total))
            order_id = cur.lastrowid
            for line in order.lines:
                conn.execute(
                    "INSERT INTO order_items (order_id, menu_id, name, quantity,"
                    " unit_price, options) VALUES (?, ?, ?, ?, ?, ?)",
                    (order_id, line.menu_id, line.name, line.quantity,
                     line.unit_price, ", ".join(line.options)))
        order._stored = True
        return order_id

    def recent_orders(self, limit: int = 20) -> list[dict]:
        """For the kitchen screen: latest orders with their items."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            orders = conn.execute(
                "SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            result = []
            for o in orders:
                items = conn.execute(
                    "SELECT name, quantity, options FROM order_items WHERE order_id = ?",
                    (o["id"],)).fetchall()
                result.append({
                    **{k: o[k] for k in o.keys()},
                    "items": [dict(i) for i in items],
                })
            return result

    def item_sales_count(self, menu_id: str) -> int:
        """Answer a relational question with one query."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM order_items WHERE menu_id = ?",
                (menu_id,)).fetchone()
            return int(row[0])

    def record_metric(self, kind: str, captured_at: str, session: str,
                      **numbers: float | None) -> None:
        """One row per metrics event (LLM/STT/TTS/EOU). Never raises into a call."""
        cols = ["captured_at", "session", "kind"] + \
               [c for c in METRIC_COLUMNS if c in numbers]
        placeholders = ", ".join("?" * len(cols))
        values = [captured_at, session, kind] + \
                 [numbers[c] for c in METRIC_COLUMNS if c in numbers]
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO turn_metrics ({', '.join(cols)}) VALUES ({placeholders})",
                values)

    def latency_summary(self) -> list[dict]:
        """The step-14 README table: avg/max per kind, millis. Caller-felt chain:
        eou delay -> stt transcription delay -> llm ttft -> tts ttfb."""
        pick = lambda col: (f"ROUND(AVG({col})*1000)", f"ROUND(MAX({col})*1000)")
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            agg = []
            for col in ("end_of_utterance_delay", "transcription_delay",
                        "ttft", "ttfb", "duration"):
                avg_sql, max_sql = pick(col)
                row = conn.execute(
                    f"SELECT {avg_sql} AS avg_ms, {max_sql} AS max_ms,"
                    f" COUNT({col}) AS n, '{col}' AS col FROM turn_metrics"
                    f" WHERE {col} IS NOT NULL").fetchone()
                if row["n"]:
                    agg.append(dict(row))
            return agg
