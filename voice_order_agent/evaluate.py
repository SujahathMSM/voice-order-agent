"""Step 12: the evaluation harness.

Two runners:
- --text: drive the text agent (same rules, same tools) - fast, no mic, tests the brain
- --live: print each scenario for the human to SPEAK in Test in Console, then
  score the latest confirmed order from SQLite - tests the whole voice chain

Scoring is a pure function over an Order (or DB row rewritten as one).

Run:
  uv run python -m voice_order_agent.evaluate --text
  uv run python -m voice_order_agent.evaluate --text --limit 3
  uv run python -m voice_order_agent.evaluate --live
"""
from .agent import OrderAgent
from .eval_scenarios import SCENARIOS
from .order import Order
from .store import OrderStore


def score_order(order: Order, expected: dict) -> tuple[bool, list[str]]:
    """Pure scorer: expected items/pickup vs the actual (confirmed) Order."""
    problems: list[str] = []
    if order.status != "confirmed":
        problems.append(f"order not confirmed (status={order.status})")
    # sum quantities per menu_id (merge makes this 1 line, but sum is robust)
    actual: dict[str, dict] = {}
    for line in order.lines:
        entry = actual.setdefault(line.menu_id, {"qty": 0, "options": set()})
        entry["qty"] += line.quantity
        entry["options"] |= set(line.options)
    for menu_id, exp in expected.get("items", {}).items():
        if menu_id not in actual:
            problems.append(f"missing item {menu_id}")
            continue
        if actual[menu_id]["qty"] != exp.get("qty", 1):
            problems.append(
                f"{menu_id}: expected qty {exp.get('qty', 1)},"
                f" got {actual[menu_id]['qty']}")
        exp_options = set(exp.get("options", []))
        if exp_options and exp_options != set(actual[menu_id]["options"]):
            problems.append(
                f"{menu_id}: expected options {sorted(exp_options)},"
                f" got {sorted(actual[menu_id]['options'])}")
    for menu_id in actual:
        if menu_id not in expected.get("items", {}):
            problems.append(f"unexpected item {menu_id}")
    exp_pickup = expected.get("pickup")
    if exp_pickup and order.pickup_time != exp_pickup:
        problems.append(f"pickup: expected {exp_pickup}, got {order.pickup_time}")
    return (not problems), problems


def run_text(only: str = "", limit: int = 0) -> None:
    """Text-mode eval. Uses the LLM - costs API calls, run --limit for smoke.
    Writes to a TEMP database: eval must never pollute the real orders.db."""
    import tempfile
    from pathlib import Path
    from .config import load_settings
    # the closer guarantee: the agent is required (rule 6) to ASK before
    # confirming; a scripted caller must answer that ask, so every scenario
    # gets these final turns appended. Without them a well-behaved agent
    # ends every call politely unconfirmed (seen in eval run 1: 6 scenarios
    # failed on 'not confirmed' with perfect items).
    CLOSER = ["Please confirm my order now.", "Yes.", "Yes, confirm it."]
    results: list[tuple[str, str, bool, list[str]]] = []
    picked = [s for s in SCENARIOS if not only or s["id"] == only][:limit or None]
    for scenario in picked:
        print(f"\n--- {scenario['id']} ({scenario['category']}) ---")
        with tempfile.TemporaryDirectory() as tmp:
            agent = OrderAgent(load_settings(),
                               store=OrderStore(db_path=Path(tmp) / "eval.db"))
            try:
                for turn in scenario["turns"] + CLOSER:
                    if agent.order.status == "confirmed":
                        break   # closers exist to reach this point - stop there
                    try:
                        reply = agent.chat(turn)
                    except Exception:                       # noqa: BLE001
                        # one retry for transient network flakes (SSL EOF etc.)
                        print(f"  (transient error on turn, retrying once)")
                        reply = agent.chat(turn)
                    print(f"  caller> {turn}")
                    print(f"  agent> {reply.strip()[:110]}")
            except Exception as error:                              # noqa: BLE001
                results.append((scenario["id"], scenario["category"], False,
                                [f"runner error: {error}"]))
                continue
        ok, problems = score_order(agent.order, scenario["expected"])
        results.append((scenario["id"], scenario["category"], ok, problems))
        print("PASS" if ok else f"FAIL: {problems}")
    _report(results)


def run_live() -> None:
    """Live-mode eval: you speak each scenario in Test in Console, then it
    scores the LATEST confirmed order found in the database."""
    store = OrderStore()
    scenario_id = input(
        "Scenario id to score (e.g. S01) after you complete the call"
        " - Enter to quit: ").strip()
    while scenario_id:
        scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
        if scenario is None:
            print(f"unknown scenario {scenario_id}")
        else:
            print(f"\nSpeak these turns, then confirm:\n  "
                  + "\n  ".join(scenario["turns"]))
            input("Press Enter when the call is done and confirmed... ")
            latest = store.recent_orders(limit=1)
            if not latest:
                print("FAIL: no confirmed order in the database")
            else:
                row = latest[0]
                flat = Order()
                flat.status = "confirmed"
                flat.pickup_time = row["pickup_time"]
                from .order import OrderLine
                for item in row["items"]:
                    # order_items stores menu_id, so live scoring uses the
                    # same ids as the text runner. (The old name-map had a
                    # Samosa/Samosas bug and was needless indirection.)
                    flat.lines.append(OrderLine(
                        1, item["menu_id"], item["name"], 0,
                        item["quantity"], (item.get("options") or "").split(", ")))
                ok, problems = score_order(flat, scenario["expected"])
                print(f"(scored DB order #{row['id']})")
                print("PASS" if ok else f"FAIL: {problems}")
        scenario_id = input(
            "\nNext scenario id to score (Enter to quit): ").strip()


def _report(results: list) -> None:
    total = len(results)
    passed = sum(1 for r in results if r[2])
    print(f"\n=== EVALUATION: {passed}/{total} scenarios passed "
          f"({100 * passed / total:.0f}%) ===")
    by_cat: dict[str, list[bool]] = {}
    for _, cat, ok, _ in results:
        by_cat.setdefault(cat, []).append(ok)
    for cat, oks in by_cat.items():
        print(f"  {cat}: {sum(oks)}/{len(oks)}")
    for sid, _, ok, problems in results:
        if not ok:
            print(f"  FAILED {sid}: {problems}")


def main() -> None:
    import sys
    args = sys.argv[1:]
    only = ""
    limit = 0
    for flag in ("--limit", "--only"):
        if flag in args:
            i = args.index(flag)
            if flag == "--limit":
                limit = int(args[i + 1])
            else:
                only = args[i + 1]
    if "--live" in args or "--text" not in args:
        run_live()
    else:
        run_text(only=only, limit=limit)


if __name__ == "__main__":
    main()
