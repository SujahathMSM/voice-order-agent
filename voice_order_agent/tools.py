"""Step 4a: the LLM-facing tools + execute_tool. The LLM is a guest, not an owner."""
import json
from typing import Callable

from .menu_search import MenuSearch, load_menu
from .order import Order

MENU = load_menu()  # id -> item dict, loaded once

# What the LLM sees: names + argument descriptions. Nothing else.
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": "Search the menu by what the caller said. Use when they ask "
                           "about items, prices, ingredients, or when a dish name is unclear. "
                           "Returns the top matches with ids, prices and details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the caller said/asked about."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add a menu item to the order. Use only menu ids that came "
                           "from search_menu results or the caller confirmed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "menu_id": {"type": "string", "description": "Menu item id, e.g. 'roll-fish'."},
                    "quantity": {"type": "integer", "description": "How many. Default 1."},
                    "options": {"type": "array", "items": {"type": "string"},
                                "description": "Modifiers like 'extra hot', 'coke'."},
                },
                "required": ["menu_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_item",
            "description": "Change the quantity or add options on an existing order line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "integer", "description": "The line number in the order."},
                    "quantity": {"type": "integer", "description": "New quantity (at least 1)."},
                    "add_options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["line_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": "Remove an order line entirely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "line_id": {"type": "integer", "description": "The line number to remove."},
                },
                "required": ["line_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_menu",
            "description": "Filter the WHOLE menu by exact fields. Use for questions like "
                           "'anything without onions?', 'what is vegetarian?', 'nothing spicy'. "
                           "Returns all matching items. Do not use search_menu for these.",
            "parameters": {
                "type": "object",
                "properties": {
                    "no_onions": {"type": "boolean", "description": "Only items without onions."},
                    "veg_only": {"type": "boolean", "description": "Only vegetarian items."},
                    "max_spicy": {"type": "integer", "description": "Only items with spicy <= this (0-3)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order",
            "description": "Lock the order after reading it back and the caller agreed. "
                           "No changes possible after this.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolBox:
    """Wires the tools to one Order + one MenuSearch. Owns execute_tool."""

    def __init__(self, order: Order, searcher: MenuSearch) -> None:
        self.order = order
        self.searcher = searcher
        self.handlers: dict[str, Callable[[dict], str]] = {
            "search_menu": self._search_menu,
            "add_item": self._add_item,
            "update_item": self._update_item,
            "remove_item": self._remove_item,
            "filter_menu": self._filter_menu,
            "confirm_order": self._confirm_order,
        }

    def execute(self, name: str, args: dict) -> str:
        """Run one tool call. ALWAYS returns text: errors become messages for the LLM."""
        tool = self.handlers.get(name)
        if tool is None:
            return f"Unknown tool: {name}"
        try:
            return tool(args)
        except (ValueError, RuntimeError) as e:
            # Exceptions kill the conversation. Text lets the LLM recover gracefully.
            return f"ERROR: {e}"

    def _search_menu(self, args: dict) -> str:
        results = self.searcher.search(args["query"], k=3)
        if not results:
            return "No menu matches found. Ask the caller to describe the dish."
        return json.dumps(
            [{"id": item["id"], "name": item["name"], "price": item["price"],
              "spicy": item["spicy"], "contains_onions": item["contains_onions"],
              "options": item["options"], "score": round(score, 2)}
             for item, score in results],
            ensure_ascii=False)

    def _add_item(self, args: dict) -> str:
        menu_id = args["menu_id"]
        item = MENU.get(menu_id)
        if item is None:
            return (f"ERROR: unknown menu id '{menu_id}'. "
                    f"Call search_menu first and use one of the returned ids.")
        line = self.order.add_item(
            menu_id, item["name"], item["unit"] if "unit" in item else item["price"],
            quantity=args.get("quantity", 1), options=args.get("options", []),
        )
        return f"Added line {line.line_id}: {line.quantity} x {line.name}. Current total: {self.order.total} LKR"

    def _update_item(self, args: dict) -> str:
        line = self.order.update_item(
            args["line_id"], quantity=args.get("quantity"),
            add_options=args.get("add_options"))
        return f"Updated line {line.line_id}: {line.quantity} x {line.name}" \
               + (f" ({', '.join(line.options)})" if line.options else "") \
               + f". Current total: {self.order.total} LKR"

    def _remove_item(self, args: dict) -> str:
        self.order.remove_item(args["line_id"])
        return f"Removed line {args['line_id']}. Current total: {self.order.total} LKR"

    def _filter_menu(self, args: dict) -> str:
        """Whole-catalog exact filter. No embeddings, no API, always complete."""
        items = list(MENU.values())
        if args.get("no_onions"):
            items = [i for i in items if not i["contains_onions"]]
        if args.get("veg_only"):
            items = [i for i in items if i["veg"]]
        if args.get("max_spicy") is not None:
            items = [i for i in items if i["spicy"] <= args["max_spicy"]]
        if not items:
            return "No items match that filter."
        return json.dumps(
            [{"id": i["id"], "name": i["name"], "price": i["price"], "spicy": i["spicy"]}
             for i in items],
            ensure_ascii=False)

    def _confirm_order(self, args: dict) -> str:
        return "CONFIRMED:\n" + self.order.confirm_order()
