"""Tests for Order state. Run: uv run pytest"""
from voice_order_agent.order import Order


def make_order() -> Order:
    order = Order()
    order.add_item("kottu-chicken", "Chicken Kottu", 1450, quantity=3)
    order.add_item("soda", "Soft Drink", 180, quantity=2, options=["coke"])
    return order


def test_add_and_total():
    order = make_order()
    # 3*1450 + 2*180 = 4350 + 360
    assert order.total == 4710


def test_update_quantity():
    order = make_order()
    order.update_item(1, quantity=4)          # "change that to four kottu"
    assert order.lines[0].quantity == 4
    assert order.total == 4 * 1450 + 360


def test_update_options_no_duplicates():
    order = make_order()
    order.update_item(1, add_options=["extra hot"])
    order.update_item(1, add_options=["extra hot"])
    assert order.lines[0].options == ["extra hot"]


def test_remove_second_item():
    order = make_order()
    order.remove_item(2)                      # "remove the second item"
    assert len(order.lines) == 1
    assert order.lines[0].menu_id == "kottu-chicken"


def test_confirm_locks_everything():
    order = make_order()
    order.pickup_time = "18:30"          # confirm now requires a pickup time
    order.confirm_order()
    for action in (lambda: order.add_item("roll-fish", "Fish Rolls", 260),
                   lambda: order.update_item(1, quantity=9),
                   lambda: order.remove_item(1),
                   lambda: order.confirm_order()):
        try:
            action()
            assert False, "edit after confirm should have failed"
        except RuntimeError:
            pass


def test_confirm_requires_pickup_time():
    """A phone order locked without a pickup time is half an order."""
    order = make_order()
    try:
        order.confirm_order()
        assert False, "confirm without pickup time should have failed"
    except ValueError as error:
        assert "pickup" in str(error)


def test_summary_includes_pickup_time():
    order = make_order()
    order.pickup_time = "18:30"
    assert "Pickup at 18:30" in order.summary()


def test_same_item_merges_into_one_line():
    """'1 hopper then 10 more hoppers' reads as ONE line: 11 x Egg Hoppers.
    A kitchen ticket with duplicate item lines is a bug, not a curiosity."""
    order = Order()
    order.add_item("hopper-egg", "Egg Hoppers", 250, quantity=1)
    line = order.add_item("hopper-egg", "Egg Hoppers", 250, quantity=10)
    assert len(order.lines) == 1
    assert line.quantity == 11 and line.line_id == 1
    assert order.total == 11 * 250
    assert "11 x Egg Hoppers" in order.summary()


def test_same_item_different_options_stay_separate():
    """Plain kottu and extra-hot kottu are different plates - no merge."""
    order = Order()
    order.add_item("kottu-chicken", "Chicken Kottu", 1450, options=[])
    order.add_item("kottu-chicken", "Chicken Kottu", 1450,
                   options=["extra hot"])
    assert len(order.lines) == 2


def test_cannot_confirm_empty():
    order = Order()
    try:
        order.confirm_order()
        assert False, "empty confirm should have failed"
    except ValueError:
        pass


def test_unknown_line():
    order = make_order()
    try:
        order.remove_item(99)
        assert False, "unknown line should have failed"
    except ValueError:
        pass
