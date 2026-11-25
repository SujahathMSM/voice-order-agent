"""Step 10: order persistence. Only CONFIRMED orders reach the database."""
from pathlib import Path

import pytest

from voice_order_agent.order import Order
from voice_order_agent.store import OrderStore


def confirmed_order() -> Order:
    order = Order()
    order.add_item("kottu-chicken", "Chicken Kottu", 1450, quantity=2,
                   options=["extra hot"])
    order.pickup_time = "18:30"
    order.confirm_order()
    return order


@pytest.fixture()
def store(tmp_path: Path) -> OrderStore:
    return OrderStore(db_path=tmp_path / "test_orders.db")


def test_open_order_is_never_saved(store):
    order = Order()
    order.add_item("roll-fish", "Fish Rolls", 260)
    with pytest.raises(ValueError):
        store.save_order(order, "2026-01-01T12:00:00")


def test_confirmed_order_saved_with_items_and_total(store):
    order = confirmed_order()
    order_id = store.save_order(order, "2026-01-01T12:00:00")
    assert order_id >= 1
    saved = store.recent_orders()
    assert len(saved) == 1
    assert saved[0]["total"] == 2900
    assert saved[0]["pickup_time"] == "18:30"
    assert saved[0]["items"][0]["quantity"] == 2
    assert "extra hot" in saved[0]["items"][0]["options"]


def test_double_save_blocked(store):
    order = confirmed_order()
    store.save_order(order, "2026-01-01T12:00:00")
    with pytest.raises(ValueError):
        store.save_order(order, "2026-01-01T12:01:00")


def test_relational_query_counts_item_sales(store):
    a = confirmed_order()          # 2 chicken kottu
    b = Order()
    b.add_item("kottu-chicken", "Chicken Kottu", 1450)
    b.pickup_time = "19:00"
    b.confirm_order()
    store.save_order(a, "2026-01-01T12:00:00")
    store.save_order(b, "2026-01-01T12:05:00")
    assert store.item_sales_count("kottu-chicken") == 3
    assert store.item_sales_count("roll-fish") == 0


def test_recent_orders_newest_first(store):
    store.save_order(confirmed_order(), "2026-01-01T12:00:00")
    order_b = confirmed_order()
    store.save_order(order_b, "2026-01-01T13:00:00")
    ids = [o["id"] for o in store.recent_orders()]
    assert ids == sorted(ids, reverse=True)
