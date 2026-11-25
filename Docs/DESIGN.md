# Design & Decisions Log — Voice Ordering Agent

This document records WHAT we built, WHY we chose it, and the alternatives we rejected.
Updated at the end of each plan step. Line numbers drift; sections and function names are
the stable references.

---

## 1. Project overview

A voice agent for a made-up restaurant ("Spice Route Kitchen"). The customer talks in the
browser; the agent answers menu questions, builds the order with tool calls, reads the
order back, recovers when the caller is frustrated, and saves confirmed orders to a
kitchen screen.

**Stack:** LiveKit Agents + WebRTC · Mistral Voxtral (STT/TTS) · Mistral Small + Meta Muse
via LLM proxy · Mistral embeddings (`mistral-embed`) · local `.npz` vector store · SQLite (step 10).

**Architecture (planned, end state):**

```
Browser (customer mic/speaker)          Kitchen screen
        │  WebRTC (audio)                    │ HTTP polling
        ▼                                   ▼
┌─────────────────────────────────────────────────────┐
│ LiveKit Server (audio transport)                    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ Agent worker (LiveKit Agents, Python)               │
│                                                     │
│  STT: Voxtral ─▶ Agent loop (LLM + tools) ─▶ TTS    │
│                       │            │                │
│              search_menu()      Order state         │
│              (hybrid retrieval)  (add/update/       │
│                       │          remove/confirm)    │
│                       ▼                 ▼          │
│              menu.json +      SQLite (confirmed     │
│              menu_vectors.npz orders, step 10)      │
└─────────────────────────────────────────────────────┘
```

---

## 2. Step 0 — Accent test (DONE)

**Files:** `accent_test.py`, `refs.json`, `audio/`

**Decision:** test STT on real recordings BEFORE building anything. The riskiest
assumption was that Voxtral understands the founder's accent.

**Result:** Voxtral 13.25% WER vs faster-whisper-small 34.37% (10 recordings). Voxtral
selected for v1.

**Findings (each drives a later design choice):**

| Finding | Consequence |
|---|---|
| The word "kottu" breaks ASR (phonetic mangle) | Hybrid retrieval with aliases (step 2); Muse Voice Transcribe keyword biasing tested in step 13 |
| "6.30" scored as an error against "six thirty" | Evaluation must normalize numbers (step 12) |
| Both models mishearing identically = bad recording, not bad model | Re-record rule: only when BOTH systems fail the same way |
| References must match audio exactly, then stay frozen | WER measures the triple (audio, reference, model); wrong references poison results |

---

## 3. Project structure & configuration (Step 1, DONE)

**Files:** `voice_order_agent/` (package), `voice_order_agent/config.py`, `.env`,
`PLAN.md`

**Decisions:**

1. **Package folder `voice_order_agent/`** (underscores) alongside the hyphenated project
   folder. Hyphens can't be imported in Python. Alternatives (flat root scripts) rejected:
   steps 5/10/12 bring entry points that would force a painful import rename.
2. **`config.py` is the ONLY module that reads `.env`/environment.** Everything else
   imports `load_settings()`. One place to change; secrets never leak into logic.
   - `load_dotenv()` anchors on `Path(__file__).parent.parent/.env`, NOT the current
     working directory — config loading is CWD-independent.
   - `load_settings()` raises `RuntimeError` with a fix-it message if the key is missing,
     instead of letting a confusing 401 surface later. Fail loudly, fail early.
3. **`.env` in `.gitignore` from day one.** Committed secrets are leaked secrets; git
   history is append-only, so a committed key means rotating the key.
4. **Strict rule learned from a real bug:** the `mistralai` SDK restructured; the class
   moved from `from mistralai import Mistral` to `from mistralai.client import Mistral`
   (now a namespace package, `mistralai.__file__ is None`). Rule: never guess library
   APIs — check docs and `dir()`/`pkgutil.iter_modules` on the installed version.

---

## 4. Menu + retrieval (Step 2, DONE)

**Files:** `data/menu.json`, `voice_order_agent/menu_index.py`,
`voice_order_agent/menu_search.py`

### 4.1 Menu schema (`data/menu.json`)

**Structure:** 30 items, each with `id`, `name`, `price` (LKR, integer), `description`,
`spicy` (0-3), `contains_onions`, `veg`, `options`, `aliases`.

- **`id` vs `name` separation (the machine key vs the human label):** tools reference the
  stable `id` ("kottu-chicken"); renaming a dish never breaks saved orders, tools, or
  tests. Names are ambiguous ("fried rice" matches 5 items); ids never are. Category
  prefixes in ids (kottu-, rice-, roll-) aid browsing and debugging.
- **`aliases`** — the repair vocabulary ("kiribath", "koththu", "coke"). Common
  mishearings get EXACT matches in the fuzzy layer instead of fuzzy ones.
- **`spicy` as 0-3, not boolean** — answers "is it very spicy?" without prose guessing.
- **`contains_onions` (exact fields)** — filterable questions ("anything without
  onions?") are answered by exact filters, not embeddings.
- **Everything sold is in the menu; the agent can only know/sell what's in it.**
  The agent never invents items. Menu file = ground truth.
- **Small items (rolls each-priced, drinks, tea) are deliberate** — they force quantity
  × unit-price handling, alias fast-paths, and realistic upsell turns. A happy-path-only
  demo menu hides exactly the bugs that surface first in production.
- **Drinks as one "Soft Drink" item with `options` + aliases** (coke/sprite/fanta) —
  fewer menu entries, agent asks which one. Alternative: three separate items = simpler
  search, longer menu. Chosen: options.

### 4.2 Indexing (`menu_index.py`)

- **Embed once, save to disk** (`data/menu_vectors.npz`, vectors + ids together). The
  menu is static; only caller questions need per-turn API calls.
- **One API call for all 30 texts** — `client.embeddings.create(model="mistral-embed",
  inputs=[...])`; `response.data[i]` matches `inputs[i]`.
- **Embedding text = name + aliases + options + description** — covers every way callers
  reference the item.
- **Vectors normalized at save time** so similarity at query time is one dot product
  (`vectors @ query_vec`), no length division per search.
- **`ids` saved WITH `vectors`** — vectors answer "how close", ids answer "which item";
  they can never get out of sync (same `.npz` file).

### 4.3 Hybrid search (`menu_search.py`)

**Two retrieval layers, combined with `max` score per item:**

1. **Fuzzy text (rapidfuzz, `partial_ratio`)** — free, no API, precise on names/aliases.
   First because free; "coke" → 1.00.
2. **Vector meaning (`mistral-embed`)** — 1 API call per query, survives heavy mangles
   ("seafloor flight less" → Seafood Fried Rice) and meaning queries ("something without
   onions" → Fresh Salad/Hoppers/Plain Fried Rice).

**Returns top-3, not top-1** — the LLM picks from a shortlist using conversation context;
a wrong-but-confident top-1 is worse than a shortlist.

**Known limitation, observed in testing:** a fully mangled dish name
("chicken second code" for "chicken kottu") returns generic chicken dishes; the mangle is
phonetic, not textual. **Accepted because the system fails into a QUESTION, not a wrong
order:** step-4 prompt rule — "weak search result = ask the caller, never guess", and the
step-7 read-back is the last line of defense. Follow-ups to test: Muse Voice Transcribe
keyword biasing (step 13).

**Vector scores are hints; exact fields are truth.** E.g. Club Sandwich ranked in the
"no onions" results by vector similarity, but `contains_onions: true` is authoritative —
the step-4 prompt must instruct the LLM to apply exact filters.

---

### 4.4 Framework choice: no LangChain / LangGraph (Step 4)

The agent loop is built by hand (~70 lines): model → tool calls → results → model →
answer. Reasons:

1. Learning goal — frameworks hide the exact concepts being learned (tool loop, message
   history, tool-result round trip).
2. Our flow is one simple loop with 5 tools; LangGraph pays off in complex branching
   multi-agent workflows with checkpointing, which we don't have.
3. LiveKit Agents already provides the runtime framework in the voice step; a third
   framework between LiveKit and our loop would multiply dependencies and debugging
   surface.

Where the frameworks WOULD win: rapid client prototypes, teams already fluent in them,
multi-agent routing with checkpoint/retry graphs. The loop being hand-built and unit
tested (`tests/test_tools.py`) is a feature of this project, not an oversight.

---

## 5. Order state & tools (Step 3, DONE)

**Files:** `voice_order_agent/order.py`, `tests/test_order.py`

**Principle: the LLM never touches order data directly.** All mutations go through small
tools (`add_item`, `update_item`, `remove_item`, `confirm_order`) that validate
everything. Bugs here are found by unit tests, not live phone calls.

| Tool | Caller phrase it serves | Safety inside |
|---|---|---|
| `add_item` | "two fish rolls, make one extra hot" | qty ≥ 1; price frozen from menu |
| `update_item` | "change that to four" | qty < 1 rejected (that's a removal); option dedupe |
| `remove_item` | "remove the second item" | unknown id → `ValueError` |
| `confirm_order` | "that's right" | refuses empty orders; LOCKS the order |

Design decisions:

- **`line_id` 1, 2, 3 in insertion order** — callers count by position ("the second one");
  the LLM maps that to `line_id` trivially.
- **Prices frozen into lines at add time, from `menu.json`, never from the LLM** — the
  LLM can't assume or hallucinate a price; a 6pm price change doesn't rewrite old orders.
- **`total` is computed, never stored** — stored totals go stale on edit; computed totals
  can't.
- **`_check_editable` guard on every mutation** — after `status == "confirmed"` all change
  tools raise. Defense in depth: even a hallucinated tool call can't change an order the
  kitchen already received.
- **`summary()` generates the read-back inside the data layer** — the LLM asks to confirm;
  the data speaks. Kills "agent forgot to mention an item" bugs: the read-back IS the
  order state, and `confirm_order()` returns `summary()` so the confirmation prompt can
  never drift from the real order.

Tests mirror the recorded accent-test phrases (q05 "change to four" →
`update_item(1, quantity=4)`; q09 → `remove_item(2)`): the eval recordings double as unit
test scenarios.

---

## 5a. Text agent (Step 4, DONE)

**Files:** `voice_order_agent/agent.py`, `voice_order_agent/tools.py`,
`tests/test_tools.py` (14/14 green)

**The loop (hand-built, no framework):** user text → `chat.complete(model, messages,
tools=TOOLS_SPEC)` → if `tool_calls`: execute each via `ToolBox.execute()` (errors become
TEXT, never exceptions), append results as `role="tool"` messages, loop (max 4 rounds) →
final assistant text. Escape valve: after 4 rounds, one call without tools forces an
answer.

**Design rules:**

- **LLM as guest, not owner:** `add_item` receives only `menu_id`/`quantity`/`options`;
  name and price are fetched from `menu.json` in Python. The LLM cannot invent prices.
- **Tool errors return as text** so the LLM can apologize and recover; unhandled
  exceptions would kill the conversation mid-call.
- **Tool results are compact JSON** (id, name, price, exact fields, score) — cheap tokens,
  fast turns.
- **`FakeSearch` in tests** — `ToolBox` accepts any object with `.search()`, so the agent
  logic is unit-testable with no API, no cost, deterministic.
- **System prompt encodes the findings** of steps 0–2: ask-don't-guess on weak search
  results; exact fields truth over vector hints; read-back before confirm; drinks upsell;
  "will be SPOKEN aloud" formatting rules (ready for step 5 with ~0 changes).

**Live-session bug (the step's main lesson):** the agent claimed "everything on the menu
contains onions" — false; 6 items are onion-free. Root cause: `search_menu` (similarity,
top-3 samples) was the only lens for a whole-catalog question, and the model generalized
from 3 samples to 30 dishes. **Fix: two question types → two tools.** Similarity
questions ("what is X") → `search_menu`; filter questions ("anything without onions")
→ new `filter_menu` (`no_onions`, `veg_only`, `max_spicy` — exact, catalog-wide, free).
After the fix the agent listed all 6 onion-free items. General lesson: an agent is smart
inside the box you build; when it's wrong, inspect the toolbox before blaming the model.
Wrong-but-confident knowledge claims come from tools that show samples where the question
needed a census.

**Verified live:** typo tolerance ("theree chciken kottu"), alias hits ("normal fried
rice" → Plain Fried Rice), positional update ("change to 4 instead of 3"), second-item
removal, read-back then confirm, drinks upsell post-confirmation.



## 5b. Voice pipeline — LiveKit + Voxtral (Step 5, DONE)

**Files:** `voice_order_agent/voice_agent.py`

**Architecture swap:** LiveKit's `AgentSession` replaces our hand-rolled step-4 loop
(turn detection, audio plumbing, interruptions come with it). Our `ToolBox`, `Order`,
menu RAG and system prompt survive unchanged; tools ported via `@function_tool`
wrappers around `ToolBox.execute` (thin: same safety code, same 14 tests).

**Key wins discovered in research:**

- Official `livekit-plugins-mistralai`: Voxtral STT (`voxtral-mini-transcribe-realtime-3`,
  streaming, server-side endpointing) + Voxtral TTS (`voxtral-mini-tts-3`) + Mistral LLM.
- **STT `context_bias`**: ~60 menu names + aliases fed to the recognizer — the kottu
  problem fixed at the source. Vocabulary repair now has TWO layers with different
  coverage: STT bias (fixes before text exists, catches phonetic mangles) and fuzzy
  search (repairs what slips through, catches typos/aliases).
- `use_realtime` TTS streams speech as the LLM generates (latency win) — see quirk below.

**Console-mode findings (smoke test):**

- Session verified: real mic speech → correct STT → tool calls → audible replies.
- Console/deprecated-mode quirks accepted, not debugged: typed turns reply without
  audio; STT websocket dies after 30s of typed-only silence; `lk cloud env` doesn't
  exist. Speaking is the product.

**Browser session via LiveKit Cloud ("Test in Console", Step 5 result — DONE):**

- Full spoken order placed AND confirmed end-to-end with the founder's accent: add,
  parallel double add, update "change that to four", onion-free filter (all 6 correct),
  read-back, confirm, lock. `agent_name="order-taker"` registered on the cloud.
- **THE KOTTU TEST, live result:** 1st attempt STT heard "three chicken coke in one
  cup"; the agent did not guess — search returned ambiguity, agent asked "Did you mean
  a soft drink with Coke?". 2nd attempt STT (with `context_bias`) heard "chicken kottu"
  correctly. The exact failure predicted in step 2, handled exactly as designed:
  failure degrades into a QUESTION, and the bias layer fixes the repeat.
- **Windows native crash root-caused** (no Python traceback -> native code; crash at
  session audio start -> race): livekit/rust-sdks#1614, soxr FFT-cache init race in
  `livekit_ffi.dll`. Fix: `warm_audio_resamplers()` runs resamplers once,
  single-threaded, in `entrypoint` before session audio (issue's verified workaround,
  100/100 clean). Original C++ assertion popup (FFT_LEN == -1, Abort/Retry/Ignore)
  blocks then kills the process — why the dev terminal "closed suddenly".
- **Latency notes:** `use_realtime=False` (set during console debugging) shows "flush
  audio emitter due to slow audio generation" on most turns → flipped back to
  `use_realtime=True` for streaming TTS; re-test. Lazy-first-import stalls
  (~1-6s once per process for mistralai) documented; prewarm hooks exist if needed.
- Session recording is enabled (`enable_recording: true`) by the console — step 12
  evaluation gets transcripts + audio from LiveKit Cloud for free.



## 5c. Barge-in / interruptions (Step 6, DONE)

**Verified live, on evidence:** interruption mid-reply detected (probability ~0.71,
agent dropped the rest of the sentence and answered the cut-in); a cough correctly
produced NO interruption; agent speech self-interruption risk handled by the AEC
warmup window (interruptions disabled 3s while greeting starts).
**Honest scope note:** AgentSession implements barge-in by default — our work was
verification and reading logs, not building it. Frameworks give the 90%; the last
10% (mic, room, thresholds) is empirical tuning per deployment.

**Two real bugs found in the same session (fixed, tested):**

1. **The phantom add.** LLM said "Two Fish Rolls added" with NO add_item call —
   a fabrication from conversation memory. The new `read_order` tool (order.summary()
   piped to the LLM) exposed the truth ("Your order is empty"), the caller pushed
   back, and the conversation repaired itself — failure degraded into a correction
   instead of a wrong confirmed order. Also pinned with prompt rule 9: "if you want
   an item on the order you MUST call add_item in this same reply; never say
   'added' for what a tool has not returned". Lesson: an LLM with order-*changing*
   tools only will narrate the order from memory; give it a tool to *consult* the
   truth, then make the read-back flow go through it.
2. **Event-loop blocking.** Our sync embeddings API call ran on the async audio
   loop: ~900ms frozen (audio, interruption checks) per search — LiveKit warned
   with the exact fix. `search_menu` wrapper now uses `asyncio.to_thread`. Lesson:
   in a voice agent, blocking the event loop = the caller hears dead air; network
   I/O inside tools goes to a thread, always.

**Also:** options are now validated against `menu.json` in `_add_item`
(the menu owns the legal option list, not the LLM) — and twice I mis-called
"hallucination" what was actually legal data (kottu's "plain" option, seafood
rice's "no onion"): lesson #10 below.

---

## 5d. Read-back hardening + pickup time (Step 7, DONE)

**Design:** a phone order is not done until it has items + a pickup time + an agreed
read-back. So pickup is DATA, not chat: `Order.pickup_time` field, `set_pickup_time`
tool normalizes spoken time ("6:30 PM" / "18:30" -> HH:MM 24h; garbage -> ERROR text
with recovery suggestion), and `confirm_order` raises without it — same
data-layer-over-prompt principle as the step-3 lock. `summary()` appends
"Pickup at HH:MM" so every read_order repeats it.

**Live evidence:**
- Gate fired correctly: "confirm the order" with no time -> agent ASKED for pickup.
- Rule 10 proved itself on a garbled time ("9.70"): repeated back as a question
  ("Did you mean 7:00 PM?"), caller confirmed, then locked. No blind writes.
- Found in the same session, fixed in prompt rules 5/6/12: (a) PM was ASSUMED from
  "6.30" — morning vs evening must be asked, never guessed; (b) a call ended with
  the order still open (drifted to drinks + goodbye) — rule 6 now forces the confirm
  loop; (c) "rupees" was spoken instead of LKR.

**Test note (teachable failure):** adding the pickup gate broke
`test_edit_after_confirm_is_text_not_crash` — its setup relied on "confirm always
locks". A refused confirm now leaves the order editable (safe failure mode); the
test's setup was stale, not the product behavior. Pinned the new semantics
(`status == "open"` after refused confirm) in `test_confirm_blocks_without_pickup_time`.

## 5e. Context & history management (Step 8, DONE)

**Problem:** LLMs have no memory — every call re-sends the whole conversation;
long calls mean slow turns, cost, and stale search results lingering in context.

**Why aggressive trimming is safe HERE (step-3 payoff):** the order lives in the
`Order` object and prices in `menu.json` — chat history is decoration. The only
hazard is cutting an assistant tool_call away from its tool results (orphaned
answers confuse the model).

**Design:** `prune_messages()` — pure, testable, no client. keep=24; system prompt
always kept; cut point advances to the next USER message (an exchange boundary);
if none exists in range, refuse to cut. Failure direction chosen deliberately:
oversized context (annoying, costs tokens) beats corrupt context (a bug).
No cut is ever taken mid-exchange; tests pin all three properties.

**Voice side decision:** NOT pruned — longest real call so far is ~9 turns, and
LiveKit exposes `session.update_chat_ctx` if evaluation ever shows degradation.
No optimization without a measurement.

## 5f. Frustration-aware recovery (Step 8, DONE)

**Design split:** detection + adaptation = PROMPT (behavior), evidence = TOOL
(counter in ToolBox, per-call; logs show `note_frustration` with reason; step 12
eval can count struggles per call). Rule 13: triggers ("what?", "sorry?", "repeat",
slow-down request, two corrections) -> slow mode for the REST of the call: short
sentences, one item per sentence, one question at a time, per-item confirm.

**Live verdict:** PASS. First trigger detected and logged; slow mode engaged
("Sorry. One chicken fried rice. Okay?") and HELD through confirm; correction
recovery ("make the rice too" misheard -> caller corrected -> quantity 2) handled.

**New bug surfaced (the pass's gift):** in slow mode the agent read the order
back FROM MEMORY — no read_order call — and did its own arithmetic: spoken total
3060 vs real 3370 LKR (verified: 2x1250 + 2x260 + 350 = 3370). Caller confirmed
on a wrong number. LLMs cannot be trusted with arithmetic — it's a fact about
the model class, not a rule-following issue. Fix: rule 5 made absolute —
"NEVER do order math yourself... the number in read_order is the only total",
applied to every read-back, not just pre-confirm. Read-back-before-confirm now
means "read-back" = tool call, every single time.

## 6. Track record of lessons that generalise

1. Verify library APIs against the installed version, never from memory (mistralai import
   restructure).
2. References/ground truth must match reality exactly, then stay frozen (WER refs).
3. Structure and config ownership decided BEFORE the code grows (package + one config module).
4. A demo dataset sized only for the happy path hides the bugs that surface first (small items in menu).
5. Data-layer guarantees beat prompt-layer hopes (locked orders, generated read-back).
6. Design the failure path, not just the success path (mangled kottu → agent asks → fuzzy fix).
7. When the agent is confidently wrong, inspect the toolbox before blaming the model
   ("everything has onions" = sample tool answering a census question).
8. An agent turn is a LOOP of API calls (1–3+ LLM calls + embedding calls per turn), not
   one call — the escape valve and history growth both follow from this.
9. Crash triage heuristics: no Python traceback = look at native code; intermittent
   crash at session start = suspect a race; a vendor bug report with a verified
   workaround beats a week of blind debugging (rust-sdks#1614).
10. Before diagnosing "the LLM hallucinated", check the data: two suspected
    hallucinations this project ("plain" option, "no onion" question) were both
    legal menu fields. Wrong-tool and invented-fact bugs are real (see #7's onion
    census and the phantom add), but verify against ground truth first, always.
11. Never let an LLM do arithmetic in production, even "easy" sums. The
    frustration-test agent summed 2x1250 + 2x260 + 350 as 3060 (real: 3370) and
    the caller confirmed a wrong total. Numbers spoken to users must come from
    code (read_order), same as numbers stored (prices): tools own all numbers.

---

## 7. Upcoming decisions already reserved

- **Step 5:** Voxtral streaming STT availability must be verified on La Plateforme; if
  batch-only, VAD-based turn detection fires transcripts only after the caller stops.
- **Step 4 prompt rules (known already):** (a) weak search = ask, never guess; (b) exact
  fields (`contains_onions`, `spicy`) are truth over vector hints; (c) read-back before
  confirm; (d) offer drinks turn after food order.
- **Step 9/13:** Muse Voice Transcribe keyword biasing; LLM proxy shape (OpenAI-compatible
  with tool calling) to be confirmed.
- **Step 12:** number normalization in scoring ("six thirty" ≡ "6:30" ≡ "18:30").
- **Step 14:** deployment (WebRTC UDP ports, TLS for mic access, config via `.env`).
