"""Step 8: history pruning must never orphan a tool result from its call."""
from voice_order_agent.agent import prune_messages


def msg(role, content="x", **extra):
    return {"role": role, "content": content, **extra}


def test_short_history_untouched():
    history = [msg("system"), msg("user", "hi"), msg("assistant", "hello")]
    assert prune_messages(history, keep=24) is history


def test_prune_keeps_system_and_cuts_at_user_boundary():
    history = [msg("system", "rules")]
    for i in range(20):
        history += [msg("user", f"u{i}"), msg("assistant", f"a{i}")]
    out = prune_messages(history, keep=8)
    assert out[0]["role"] == "system"
    assert len(out) <= 9
    assert out[1]["role"] == "user"          # cut landed on a user message


def test_prune_never_splits_tool_call_pair():
    """If the only safe cut would orphan tool results, cut later instead."""
    history = [msg("system", "rules")]
    for i in range(10):
        history += [msg("user", f"u{i}"), msg("assistant", f"a{i}")]
    # a tool exchange with NO user message after it yet
    history += [
        msg("assistant", None, tool_calls=[
            {"id": "call_1", "type": "function",
             "function": {"name": "search_menu", "arguments": "{}"}}]),
        {"role": "tool", "tool_call_id": "call_1", "name": "search_menu",
         "content": "{}"},
    ]
    out = prune_messages(history, keep=8)
    # find any tool result kept -> its call must be kept too
    kept_ids = {tc["id"] for m in out for tc in m.get("tool_calls", [])}
    for m in out:
        if m["role"] == "tool":
            assert m["tool_call_id"] in kept_ids, "orphaned tool result!"


def test_no_safe_boundary_means_no_cut():
    history = [msg("system", "rules"), msg("assistant", "hi"),
               msg("tool", "result", tool_call_id="c1")]
    assert prune_messages(history, keep=2) == history   # refuses to cut
