"""Step 11: latency metrics storage and aggregation."""
from pathlib import Path

import pytest

from voice_order_agent.store import OrderStore


@pytest.fixture()
def store(tmp_path: Path) -> OrderStore:
    return OrderStore(db_path=tmp_path / "metrics.db")


def test_record_and_summarize(store):
    store.record_metric("llm", "t1", "s1", duration=0.9, ttft=0.5,
                        cancelled=0, prompt_tokens=800, completion_tokens=40)
    store.record_metric("llm", "t2", "s2", duration=1.1, ttft=0.7,
                        cancelled=0, prompt_tokens=900, completion_tokens=50)
    store.record_metric("tts", "t3", "s1", duration=1.8, ttfb=0.4,
                        audio_duration=3.2, cancelled=0)
    store.record_metric("eou", "t4", "s1", end_of_utterance_delay=0.3,
                        transcription_delay=2.5)

    summary = {row["col"]: row for row in store.latency_summary()}
    assert summary["ttft"]["avg_ms"] == 600        # (0.5+0.7)/2 in millis
    assert summary["ttft"]["max_ms"] == 700
    assert summary["ttft"]["n"] == 2
    assert summary["ttfb"]["n"] == 1 and summary["ttfb"]["avg_ms"] == 400
    assert summary["transcription_delay"]["avg_ms"] == 2500


def test_summary_ignores_null_columns(store):
    """An LLM row has no ttfb; it must not dilute the TTS ttfb average."""
    store.record_metric("llm", "t1", "s1", duration=0.9, ttft=0.5)
    summary = {row["col"]: row for row in store.latency_summary()}
    assert "ttfb" not in summary                 # only real values counted
    assert summary["ttft"]["n"] == 1
