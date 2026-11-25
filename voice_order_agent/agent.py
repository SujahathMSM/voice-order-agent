"""Step 4b: the agent loop. Model -> tools -> model -> answer, in a terminal."""
import json

from mistralai.client import Mistral

from .config import Settings, load_settings
from .menu_search import MenuSearch
from .order import Order
from .tools import TOOLS_SPEC, ToolBox

MAX_TOOL_ROUNDS = 4   # escape valve: never loop tools forever

SYSTEM_PROMPT = """You are the counter assistant at Spice Route Kitchen, a Sri Lankan
fusion restaurant. You take orders by phone. Be warm, brief and natural - this will be
SPOKEN aloud, so short sentences, no lists longer than 3 items, no markdown.

RULES (follow exactly):
1. Never guess a dish. If search_menu results are weak or several items match, ask the
   caller a short question instead of adding anything.
2. Use filter_menu for questions over the WHOLE menu (no onions, vegetarian, not
   spicy). Never guess from just search_menu results - it only shows the top 3.
3. Exact fields are truth: spicy 0-3 and contains_onions come from the tool results.
   Do not invent ingredients or prices. All prices are in LKR.
4. After any change to the order, give a very short confirmation of what changed.
5. Before calling confirm_order, ALWAYS read the full order back first and wait for
   the caller to agree.
6. Once the food order is complete, ask if they want something to drink.
7. If a tool returns ERROR: ..., tell the caller plainly and suggest the fix.
7. Keep small talk minimal. Your job: correct order, confirmed quickly.
"""


class OrderAgent:
    """One order, one conversation. The same brain will run inside LiveKit in step 5."""

    def __init__(self, settings: Settings) -> None:
        self.client = Mistral(api_key=settings.mistral_api_key)
        self.order = Order()
        self.toolbox = ToolBox(self.order, MenuSearch(settings))
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_text: str) -> str:
        """One turn: append user text, loop tools if needed, return the agent's reply."""
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=self.messages,
                tools=TOOLS_SPEC,
            )
            message = response.choices[0].message

            tool_calls = message.tool_calls or []
            if not tool_calls:                      # no tools needed: final answer
                self.messages.append({"role": "assistant", "content": message.content})
                return message.content

            # The model wants tools. Remember its decision, run every call.
            self.messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {
                        "name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = self.toolbox.execute(tc.function.name, args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result,
                })
        # Escape valve fired: force an answer with tools disabled.
        response = self.client.chat.complete(
            model="mistral-small-latest", messages=self.messages)
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


def main() -> None:
    """Terminal chat. Type 'quit' to exit."""
    agent = OrderAgent(load_settings())
    print("Spice Route Kitchen - text agent (type 'quit' to exit)")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ("quit", "exit"):
            break
        if not user:
            continue
        print("\nAgent:", agent.chat(user))


if __name__ == "__main__":
    main()
