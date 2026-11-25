"""Step 12 fix: the replacement primitive. Eval S12 showed the agent narrating
'changed to deviled chicken' without any tool that could do it."""
from voice_order_agent.order import Order


def test_replace_item_swaps_dish_keeps_quantity():
    order = Order()
    order.add_item("sandwich-deviled-fish", "Deviled Fish Sandwich", 1050, quantity=2)
    line = order.replace_item(1, "sandwich-deviled-chicken",
                              "Deviled Chicken Sandwich", 980)
    assert len(order.lines) == 1          # swap, not a second line
    assert line.menu_id == "sandwich-deviled-chicken"
    assert line.quantity == 2              # quantity untouched by default
    assert order.total == 2 * 980          # new unit price applies


def test_replace_item_can_set_quantity_and_options():
    order = Order()
    order.add_item("rice-fried-plain", "Plain Fried Rice", 900, quantity=1)
    order.replace_item(1, "rice-fried-seafood", "Seafood Fried Rice",
                       1650, quantity=3, options=["no onion"])
    assert order.lines[0].quantity == 3
    assert order.lines[0].options == ["no onion"]


def test_replace_item_locked_order_raises():
    order = Order()
    order.add_item("roll-fish", "Fish Rolls", 260)
    order.pickup_time = "18:30"
    order.confirm_order()
    try:
        order.replace_item(1, "roll-egg", "Egg Rolls", 200)
        assert False, "replace after confirm should have failed"
    except RuntimeError:
        pass
