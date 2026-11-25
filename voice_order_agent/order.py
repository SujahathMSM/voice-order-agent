"""Step 3: Order state. The ONLY code allowed to change an order."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrderLine:
    line_id: int
    menu_id: str      # stable key into menu.json (e.g. "kottu-chicken")
    name: str         # what the caller hears (e.g. "Chicken Kottu")
    unit_price: int   # in LKR, frozen from the menu at add time
    quantity: int
    options: list[str] = field(default_factory=list)

    @property
    def line_total(self) -> int:
        return self.unit_price * self.quantity


@dataclass
class Order:
    """One customer's order. The LLM will drive this ONLY via tool calls."""
    lines: list[OrderLine] = field(default_factory=list)
    status: str = "open"            # "open" or "confirmed"
    pickup_time: Optional[str] = None   # "HH:MM" 24h, set via set_pickup_time
    _next_id: int = 1

    def _check_editable(self) -> None:
        if self.status == "confirmed":
            raise RuntimeError("Order is confirmed and locked. Start a new order.")

    def add_item(self, menu_id: str, name: str, unit_price: int,
                 quantity: int = 1, options: Optional[list[str]] = None) -> OrderLine:
        """Caller ordered something new. Returns the created line.
        Same menu_id + same options MERGE into one line (1 + 10 = 11,
        never two lines) - a kitchen ticket must read as one item."""
        self._check_editable()
        if quantity < 1:
            raise ValueError("Quantity must be at least 1")
        options = options or []
        for line in self.lines:
            if line.menu_id == menu_id and sorted(line.options) == sorted(options):
                line.quantity += quantity
                return line
        line = OrderLine(self._next_id, menu_id, name, unit_price,
                         quantity, options)
        self.lines.append(line)
        self._next_id += 1
        return line

    def update_item(self, line_id: int,
                    quantity: Optional[int] = None,
                    add_options: Optional[list[str]] = None) -> OrderLine:
        """Change quantity or add options on an existing line ('make it four')."""
        self._check_editable()
        line = self.find_line(line_id)
        if quantity is not None:
            if quantity < 1:
                raise ValueError("Use remove_item instead of quantity 0")
            line.quantity = quantity
        if add_options:
            for option in add_options:
                if option not in line.options:
                    line.options.append(option)
        return line

    def remove_item(self, line_id: int) -> None:
        """Delete a line entirely ('remove the second item')."""
        self._check_editable()
        line = self.find_line(line_id)
        self.lines.remove(line)

    def find_line(self, line_id: int) -> OrderLine:
        for line in self.lines:
            if line.line_id == line_id:
                return line
        raise ValueError(f"No order line with id {line_id}")

    @property
    def total(self) -> int:
        return sum(line.line_total for line in self.lines)

    def summary(self) -> str:
        """The read-back text. Step 7 will speak this on the phone."""
        if not self.lines and not self.pickup_time:
            return "The order is currently empty."
        if not self.lines:
            return f"No items yet. Pickup at {self.pickup_time}."
        rows = [f"{line.quantity} x {line.name}"
                + (f" ({', '.join(line.options)})" if line.options else "")
                + f" - {line.line_total} LKR"
                for line in self.lines]
        text = "\n".join(rows) + f"\nTotal: {self.total} LKR"
        if self.pickup_time:
            text += f"\nPickup at {self.pickup_time}"
        return text

    def confirm_order(self) -> str:
        """Lock the order. Only allowed when it is not empty AND has a pickup time."""
        self._check_editable()
        if not self.lines:
            raise ValueError("Cannot confirm an empty order")
        if not self.pickup_time:
            raise ValueError("No pickup time set - call set_pickup_time first")
        self.status = "confirmed"
        return self.summary()
