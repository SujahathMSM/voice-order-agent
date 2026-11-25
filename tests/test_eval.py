"""Step 12: scorer is a pure function - test it without any LLM or mic."""
from voice_order_agent.evaluate import score_order
from voice_order_agent.order import Order


def make_confirmed(items, pickup="18:30") -> Order:
    """items: {(menu_id, qty, options)}"""
    order = Order()
    prices = {"kottu-chicken": 1450, "soda": 120, "roll-fish": 260}
    for menu_id, qty, options in items:
        order.add_item(menu_id, menu_id, prices.get(menu_id, 100),
                       quantity=qty, options=options)
    order.pickup_time = pickup
    order.confirm_order()
    return order


def test_exact_match_passes():
    order = make_confirmed([("kottu-chicken", 2, []), ("soda", 1, [])])
    ok, problems = score_order(order, {
        "items": {"kottu-chicken": {"qty": 2}, "soda": {"qty": 1}},
        "pickup": "18:30"})
    assert ok, problems


def test_wrong_quantity_fails():
    order = make_confirmed([("kottu-chicken", 3, [])])
    ok, problems = score_order(order, {"items": {"kottu-chicken": {"qty": 2}}})
    assert not ok and any("expected qty 2" in p for p in problems)


def test_missing_and_unexpected_items():
    order = make_confirmed([("roll-fish", 1, [])])
    ok, problems = score_order(order, {"items": {"soda": {"qty": 1}}})
    assert not ok
    assert any("missing item soda" in p for p in problems)
    assert any("unexpected item roll-fish" in p for p in problems)


def test_options_must_match():
    order = make_confirmed([("kottu-chicken", 1, ["plain"])])
    ok, problems = score_order(order, {
        "items": {"kottu-chicken": {"qty": 1, "options": ["extra hot"]}}})
    assert not ok and any("expected options" in p for p in problems)


def test_unconfirmed_order_fails():
    order = Order()
    order.add_item("kottu-chicken", "Chicken Kottu", 1450)
    order.pickup_time = "18:30"
    ok, problems = score_order(order, {
        "items": {"kottu-chicken": {"qty": 1}}, "pickup": "18:30"})
    assert not ok and any("not confirmed" in p for p in problems)


def test_merged_lines_sum_quantities():
    """1 + 10 hoppers in one merged line scores as 11."""
    order = Order()
    order.add_item("roll-fish", "Fish Rolls", 260, quantity=1)
    order.add_item("roll-fish", "Fish Rolls", 260, quantity=10)
    order.pickup_time = "18:30"
    order.confirm_order()
    ok, problems = score_order(order, {"items": {"roll-fish": {"qty": 11}}})
    assert ok, problems


def test_wrong_pickup_fails():
    order = make_confirmed([("soda", 1, [])], pickup="19:00")
    ok, problems = score_order(order, {
        "items": {"soda": {"qty": 1}}, "pickup": "18:00"})
    assert not ok and any("pickup" in p for p in problems)
