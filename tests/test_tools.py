"""Tests for ToolBox. No LLM, no API — we fake the search with a stub."""
from voice_order_agent.order import Order
from voice_order_agent.tools import ToolBox, load_menu


class FakeSearch:
    """Stands in for MenuSearch: no API, fixed results."""
    def search(self, query: str, k: int = 3):
        return [(load_menu()["roll-fish"], 0.95)]


def make_box() -> tuple[ToolBox, Order]:
    order = Order()
    return ToolBox(order, FakeSearch()), order


def test_add_item_fetches_price_from_menu():
    box, order = make_box()
    result = box.execute("add_item", {"menu_id": "roll-fish", "quantity": 2})
    assert "Fish Rolls" in result
    assert order.total == 520          # 2 x 260, price came from menu.json


def test_unknown_menu_id_is_text_not_crash():
    box, _ = make_box()
    result = box.execute("add_item", {"menu_id": "pizza-hawaii", "quantity": 1})
    assert result.startswith("ERROR:")
    assert "search_menu" in result     # error tells the LLM how to fix itself


def test_edit_after_confirm_is_text_not_crash():
    box, _ = make_box()
    box.execute("add_item", {"menu_id": "roll-fish"})
    box.execute("set_pickup_time", {"time": "18:30"})   # gate: no lock without it
    box.execute("confirm_order", {})
    result = box.execute("update_item", {"line_id": 1, "quantity": 5})
    assert result.startswith("ERROR: Order is confirmed")


def test_unknown_tool_is_text_not_crash():
    box, _ = make_box()
    assert box.execute("unknown_tool_xyz", {}).startswith("Unknown tool")


def test_read_order_reflects_real_state():
    """The lying-read-back bug from the step 6 voice test: agent claimed an
    item was added when it never was. read_order is the ground truth."""
    box, order = make_box()
    result = box.execute("read_order", {})
    assert result.startswith("The order is currently empty")
    box.execute("add_item", {"menu_id": "roll-fish", "quantity": 2})
    result = box.execute("read_order", {})
    assert "2 x Fish Rolls" in result
    assert "Chicken Kottu" not in result   # never ordered -> must not appear


def test_add_item_rejects_invented_options():
    """Only the menu decides legal options. kottu-chicken allows plain and
    extra hot; anything else must come back as an ERROR string."""
    box, order = make_box()
    result = box.execute("add_item", {"menu_id": "kottu-chicken",
                                       "quantity": 1, "options": ["pineapple"]})
    assert result.startswith("ERROR:")
    assert len(order.lines) == 0                      # nothing was added
    result = box.execute("add_item", {"menu_id": "kottu-chicken",
                                       "quantity": 1, "options": ["extra hot"]})
    assert result.startswith("Added line 1")


def test_pickup_time_normalizes_and_rejects_garbage():
    box, order = make_box()
    assert box.execute("set_pickup_time", {"time": "6:30 PM"}) == "Pickup time set to 18:30."
    assert order.pickup_time == "18:30"
    assert "18:30" in box.execute("set_pickup_time", {"time": "18:30"})
    garbage = box.execute("set_pickup_time", {"time": "tomorrow-ish"})
    assert garbage.startswith("ERROR:")


def test_confirm_blocks_without_pickup_time():
    box, order = make_box()
    box.execute("add_item", {"menu_id": "roll-fish", "quantity": 2})
    result = box.execute("confirm_order", {})
    assert result.startswith("ERROR:")           # tools catch, text not crash
    assert order.status == "open"                # gate refused: nothing locked
    box.execute("set_pickup_time", {"time": "6:30 pm"})
    result = box.execute("confirm_order", {})
    assert result.startswith("CONFIRMED:") and "Pickup at 18:30" in result


def test_search_menu_returns_json_ids():
    box, _ = make_box()
    result = box.execute("search_menu", {"query": "fish rolls"})
    assert "roll-fish" in result and "Fish Rolls" in result


def test_filter_menu_no_onions_is_complete():
    """The bug from the live session: agent said 'everything has onions'."""
    box, _ = make_box()
    result = box.execute("filter_menu", {"no_onions": True})
    assert "Fresh Salad" in result and "Plain Hoppers" in result
    assert "Chicken Kottu" not in result   # contains onions -> must be absent


def test_filter_menu_veg_and_spicy():
    box, _ = make_box()
    mild_veg = box.execute("filter_menu", {"veg_only": True, "max_spicy": 0})
    assert "Plain Hoppers" in mild_veg
    assert "Dhal Curry" not in mild_veg          # veg but spicy=1 -> excluded by max_spicy
