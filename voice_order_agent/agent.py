"""Step 4b: the agent loop. Model -> tools -> model -> answer, in a terminal."""
import json

from mistralai.client import Mistral

from .config import Settings, load_settings
from .menu_search import MenuSearch
from .order import Order
from .store import OrderStore
from .tools import TOOLS_SPEC, ToolBox

MAX_TOOL_ROUNDS = 4   # escape valve: never loop tools forever
KEEP_MESSAGES = 24    # after pruning: system prompt + last ~24 messages


def prune_messages(messages: list[dict], keep: int = KEEP_MESSAGES) -> list[dict]:
    """Trim old turns. State is safe to drop because the ORDER lives in the
    Order object, not in chat. The only hazard is cutting an assistant
    tool_call apart from its tool results, so we only cut at a USER message
    boundary; if no boundary exists in range, we don't cut at all."""
    if len(messages) <= keep:
        return messages
    start = max(1, len(messages) - keep)     # never touch the system prompt
    while start < len(messages) and messages[start]["role"] != "user":
        start += 1
    if start >= len(messages):
        return messages                      # no safe boundary: keep everything
    return [messages[0]] + messages[start:]

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
5. Before ANY read-back of the order - and before confirm_order - call read_order
   and speak ONLY its lines and total. NEVER do order math yourself: no adding
   prices, no computing totals, ever. The number in read_order is the only total.
   Agree on pickup time (call set_pickup_time), then wait for the caller to agree
   before locking. If the caller does not say AM or PM, ASK which one they mean -
   never guess morning vs evening.
6. Once the order has items AND a pickup time, keep steering to confirmation:
   read the order back and ask to confirm. Never let the call end with an order
   left unconfirmed - "is that everything, shall I confirm your order?"
7. Once the food order is complete, ask if they want something to drink.
8. If a tool returns ERROR: ..., tell the caller plainly and suggest the fix.
9. NEVER claim an item was added unless the add tool result confirms it. Never
   describe the order from memory - call read_order and speak only what it says.
10. You have NO memory of tool results beyond what the conversation shows. If you
    want an item on the order, you MUST call add_item in this same reply. Never
    say "added" or "done" as a promise for something a tool has not returned yet.
11. When you are NOT sure you heard the caller correctly (unusual dish name,
    quantity, or a transcript that looks garbled), repeat the item and quantity
    back BEFORE adding: "Two fish rolls, was that right?" Never add a guess.
    But do NOT ask when the caller already stated item AND quantity clearly -
    "one plain fried rice" gets added, not re-confirmed.
12. Always say "LKR" ("rupees" confuses prices).
15. CORRECTIONS REPLACE, THEY NEVER ADD: "make that four", "change to four",
    "I meant four", "no, four" all mean UPDATE the existing line to quantity 4
    via update_item - never add on top of what is already there.
    A correction to a DIFFERENT dish ("change the fish sandwiches to deviled
    chicken") means replace_item - never just narrate it; if you cannot make
    the change with a tool call, say so.
13. FRUSTRATION: if the caller says "what?", "sorry?", "repeat", asks you to slow
    down, OR corrects you twice in a row: call note_frustration with the reason,
    then switch to SLOW MODE for the REST of the call: very short sentences;
    ONE item per sentence; confirm each item before continuing ("Two chicken
    kottu. Okay?"); ONE question at a time, never stacked; repeat the total
    slowly at the end. Never switch back to fast mode mid-call.
14. After confirm_order succeeds, tell the caller their ORDER NUMBER: the confirm
    result starts with "ORDER #N" - say "your order number is N". They need it
    at the counter.
"""


class OrderAgent:
    """One order, one conversation. The same brain will run inside LiveKit in step 5."""

    def __init__(self, settings: Settings, store=None) -> None:
        self.client = Mistral(api_key=settings.mistral_api_key)
        self.order = Order()
        if store is None:
            store = OrderStore()
        self.toolbox = ToolBox(self.order, MenuSearch(settings), store=store)
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _snapshot_order(self):
        """Deep copy of order state: chat() is all-or-nothing, and a failed
        turn must not leave half-applied tool effects behind either."""
        import copy
        return (copy.deepcopy(self.order.lines), self.order.status,
                self.order.pickup_time, self.order._next_id)

    def _restore_order(self, snapshot) -> None:
        import copy
        (self.order.lines, self.order.status,
         self.order.pickup_time, self.order._next_id) = copy.deepcopy(snapshot)

    def chat(self, user_text: str) -> str:
        """One turn: append user text, loop tools if needed, return the agent's
        reply. ALL-OR-NOTHING: works on copies; on any failure (network flake,
        malformed tool args) neither the message history nor the order keeps
        partial effects, so a retry cannot double-add or orphan tool calls."""
        order_snapshot = self._snapshot_order()
        saved_messages = self.messages
        new_messages = prune_messages(list(self.messages))
        new_messages.append({"role": "user", "content": user_text})
        try:
            return self._chat_loop(new_messages)
        except Exception:
            self.messages = saved_messages
            self._restore_order(order_snapshot)
            raise

    def _chat_loop(self, new_messages: list) -> str:
        self.messages = new_messages
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
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = None     # malformed: let execute() return ERROR text
                if not isinstance(args, dict):
                    args = None
                result = (self.toolbox.execute(tc.function.name, args)
                          if args is not None else
                          "ERROR: malformed tool arguments; call the tool again.")
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
