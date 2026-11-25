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
