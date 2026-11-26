"""Review batch 1: one test per bug found by the external code review.
Each test names the bug it pins, so a regression has a paper trail."""
import pytest

from voice_order_agent.order import Order
from voice_order_agent.store import OrderStore
from voice_order_agent.tools import ToolBox


def make_box(store=None, tmp=None):
    """A ToolBox against the real menu, optional store."""
    from voice_order_agent.menu_search import MenuSearch
    order = Order()
    searcher = MenuSearch.__new__(MenuSearch)   # no embeddings needed: fuzzy only
    box = ToolBox(order, searcher, store=store)
    return box, order


class ExplodingStore(OrderStore):
    """Simulates a DB outage: every save fails (review bug 1)."""
    def __init__(self):   # noqa: D107 - no schema setup needed
        pass

    def save_order(self, order, confirmed_at):    # noqa: ARG002
        raise OSError("disk on fire")


# ---- Bug 1: a confirmed order can be lost --------------------------------

def test_confirm_save_failure_reverts_lock_and_is_retryable():
    box, order = make_box(store=ExplodingStore())
    box.execute("add_item", {"menu_id": "roll-fish", "quantity": 1})
    box.execute("set_pickup_time", {"time": "6:30 pm"})
    result = box.execute("confirm_order", {})
    assert result.startswith("ERROR:")           # outage -> text, not crash
    assert order.status == "open"                 # lock REVERTED...
    assert order.total == 260                     # ...and the order survives
    # retry (store healed) succeeds - nothing was half-confirmed
    box.store = OrderStore(db_path=__import__("pathlib").Path(
        __import__("tempfile").mkdtemp()) / "h.db")
    result = box.execute("confirm_order", {})
    assert result.startswith("ORDER #") and order.status == "confirmed"


# ---- Bug 2: read_order must show line ids --------------------------------

def test_read_order_shows_line_ids_after_removal():
    box, order = make_box()
    box.execute("add_item", {"menu_id": "roll-fish", "quantity": 1})     # line 1
    box.execute("add_item", {"menu_id": "roll-egg", "quantity": 2})      # line 2
    box.execute("remove_item", {"line_id": 1})
    text = box.execute("read_order", {})
    assert "Line 2" in text                  # ids survive removals
    order._check_editable()                 # still editable: sanity


# ---- Bug 3: the data layer rejects bad values on every write path --------

def test_quantity_gates_in_order_layer():
    order = Order()
    for bad in (-3, 0, 1_000_000, "4", 2.5):
        with pytest.raises((ValueError, TypeError)):
            order.add_item("roll-fish", "Fish Rolls", 260, quantity=bad)


def test_update_item_add_options_validated_against_menu():
    box, order = make_box()
    box.execute("add_item", {"menu_id": "kottu-chicken", "quantity": 1,
                             "options": ["plain"]})
    result = box.execute("update_item",
                         {"line_id": 1, "add_options": ["pineapple"]})
    assert result.startswith("ERROR:")
    assert order.lines[0].options == ["plain"]     # nothing leaked through


def test_replace_item_discards_old_dish_options():
    box, order = make_box()
    box.execute("add_item", {"menu_id": "kottu-chicken", "quantity": 1,
                             "options": ["plain"]})
    # replace with fish rolls; 'plain' belongs to kottu and must not carry over
    result = box.execute("replace_item", {"line_id": 1,
                                           "new_menu_id": "roll-fish"})
    assert not result.startswith("ERROR")
    assert order.lines[0].menu_id == "roll-fish"
    assert order.lines[0].options == []


def test_replace_item_merges_with_identical_line():
    box, order = make_box()
    box.execute("add_item", {"menu_id": "roll-fish", "quantity": 1})
    box.execute("add_item", {"menu_id": "kottu-chicken", "quantity": 2})
    box.execute("replace_item", {"line_id": 2, "new_menu_id": "roll-fish"})
    assert len(order.lines) == 1                  # not the duplicate-line bug again
    assert order.lines[0].quantity == 3


# ---- Bug 4: malformed tool arguments never crash the turn ----------------

def test_execute_survives_missing_keys_and_string_numbers():
    box, order = make_box()
    assert box.execute("add_item", {}).startswith("ERROR")           # KeyError
    assert box.execute("add_item", {"menu_id": "roll-fish",
                                    "quantity": "2"}).startswith("Added")
    assert order.lines[0].quantity == 2                             # coerced
    assert box.execute("filter_menu", {"max_spicy": "2"})            # coerced
    assert box.execute("add_item", {"menu_id": "roll-fish",
                                    "quantity": "two"}).startswith("ERROR")
    assert box.execute("add_item", {"menu_id": "roll-fish",
                                    "quantity": [1]}).startswith("ERROR")


# ---- Bug 6: pickup time parsing ------------------------------------------

def test_pickup_time_voxtral_dot_format_and_bare_hour():
    box, order = make_box()
    # '6.30 pm' is literally how Voxtral writes it (step 0 finding)
    assert box.execute("set_pickup_time", {"time": "6.30 pm"}) == \
        "Pickup time set to 18:30."
    # bare '7' is an AM/PM guess: refused (rule 5), never assumed
    assert box.execute("set_pickup_time", {"time": "7"}).startswith("ERROR")
    assert box.execute("set_pickup_time", {"time": "13 pm"}).startswith("ERROR")
    assert order.pickup_time == "18:30"


# ---- Bug 7 (live eval): DB rows score by menu_id, not name ---------------

def test_recent_orders_items_expose_menu_id(tmp_path):
    store = OrderStore(db_path=tmp_path / "db.sqlite")
    order = Order()
    order.add_item("samosa", "Samosas", 100, quantity=3)
    order.pickup_time = "18:00"
    order.confirm_order()
    store.save_order(order, "t")
    row = store.recent_orders()[0]
    assert row["items"][0]["menu_id"] == "samosa"    # id, not display name


# ---- Bug 8: bare 'koththu' resolves to chicken, not vegetable ------------

def test_no_alias_points_to_two_dishes():
    """'koththu' was an alias for both chicken and veg kottu; the searcher's
    dict kept veg last, so a bare 'koththu' caller got Vegetable by accident."""
    from pathlib import Path
    import json
    menu = json.loads((Path(__file__).resolve().parent.parent
                       / "data" / "menu.json").read_text())["items"]
    seen = {}
    for item in menu:
        for alias in item["aliases"]:
            assert alias not in seen, (
                f"alias '{alias}' points to both {seen[alias]} and {item['id']}")
            seen[alias] = item["id"]
