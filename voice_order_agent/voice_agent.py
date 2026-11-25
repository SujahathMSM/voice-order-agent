"""Step 5: the voice agent. LiveKit runs the loop; our ToolBox stays the brain."""
import asyncio
import json

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import mistralai

from .agent import SYSTEM_PROMPT
from .config import load_settings
from .menu_search import MenuSearch
from .order import Order
from .tools import MENU, ToolBox

STT_MODEL = "voxtral-mini-transcribe-realtime-3"
TTS_MODEL = "voxtral-mini-tts-3"
TTS_VOICE = "voice_447"          # American English; change if you prefer another
LLM_MODEL = "mistral-small-latest"


def context_bias_words() -> list[str]:
    """Every name + alias from the menu: STT gets told what to listen for.
    This is the kottu fix at the source (Voxtral context biasing)."""
    words: list[str] = []
    for item in MENU.values():
        for text in [item["name"]] + item.get("aliases", []):
            if text not in words:
                words.append(text)
    return words  # 30 items -> ~60 words, well under the 100 limit


def make_tools(box: ToolBox):
    """Wrap our ToolBox in LiveKit function_tools. Each docstring is what
    the LLM reads (same text as our step-4 TOOLS_SPEC descriptions)."""

    @function_tool
    async def search_menu(context: RunContext, query: str) -> str:
        """Search the menu by what the caller said. Use when they ask about
        items, prices, ingredients, or when a dish name is unclear."""
        # to_thread: the embedding API call is network I/O. Running it on the
        # event loop froze audio handling ~1s per search (seen in step 6 logs).
        return await asyncio.to_thread(box.execute, "search_menu", {"query": query})

    @function_tool
    async def filter_menu(context: RunContext, no_onions: bool = False,
                          veg_only: bool = False, max_spicy: int | None = None) -> str:
        """Filter the WHOLE menu by exact fields. Use for 'anything without
        onions?', 'what is vegetarian?', 'nothing spicy'."""
        return box.execute("filter_menu", {
            "no_onions": no_onions, "veg_only": veg_only, "max_spicy": max_spicy})

    @function_tool
    async def read_order(context: RunContext) -> str:
        """Read the current order state: every line, quantity, options and total.
        ALWAYS call before reading the order back to the caller."""
        return box.execute("read_order", {})

    @function_tool
    async def add_item(context: RunContext, menu_id: str,
                       quantity: int = 1, options: list[str] | None = None) -> str:
        """Add a menu item to the order. Use only menu ids from search_menu
        results or that the caller confirmed."""
        return box.execute("add_item", {"menu_id": menu_id, "quantity": quantity,
                                         "options": options or []})

    @function_tool
    async def update_item(context: RunContext, line_id: int,
                         quantity: int | None = None,
                         add_options: list[str] | None = None) -> str:
        """Change the quantity or add options on an existing order line."""
        return box.execute("update_item", {"line_id": line_id, "quantity": quantity,
                                           "add_options": add_options})

    @function_tool
    async def remove_item(context: RunContext, line_id: int) -> str:
        """Remove an order line entirely."""
        return box.execute("remove_item", {"line_id": line_id})

    @function_tool
    async def set_pickup_time(context: RunContext, time: str) -> str:
        """Set when the caller will pick up. Normalize what they said into
        clock time, e.g. '6:30 PM' or '18:30'. Call before confirm_order."""
        return box.execute("set_pickup_time", {"time": time})

    @function_tool
    async def note_frustration(context: RunContext, reason: str) -> str:
        """Call ONCE when you detect the caller struggling ('what?', 'repeat',
        asks to slow down, corrects you twice). Give a short reason. After this,
        continue in slow step-by-step mode for the rest of the call."""
        return box.execute("note_frustration", {"reason": reason})

    @function_tool
    async def confirm_order(context: RunContext) -> str:
        """Lock the order. Requires items AND a pickup time. Only call after
        reading the order back and the caller agreed."""
        return box.execute("confirm_order", {})

    return [search_menu, filter_menu, add_item, update_item, remove_item,
            read_order, set_pickup_time, note_frustration, confirm_order]


def warm_audio_resamplers() -> None:
    """Workaround for livekit/rust-sdks#1614: the soxr FFT cache inside
    livekit_ffi.dll initializes lazily WITHOUT a lock. If two resamplers are
    first-used at the same time (mic in + TTS out at session start), the
    half-initialized cache trips 'Assertion failed: FFT_LEN == -1' and Windows
    kills the whole process. Fix per the issue: force the cache init once,
    single-threaded, before any session audio flows."""
    import numpy as np
    from livekit import rtc

    for in_rate, out_rate in [(48000, 16000), (24000, 48000),
                              (44100, 24000), (48000, 24000)]:
        resampler = rtc.AudioResampler(in_rate, out_rate)
        silence = np.zeros(480, dtype=np.int16).tobytes()
        frame = rtc.AudioFrame(silence, in_rate, 1, 480)
        for _ in range(5):
            resampler.push(frame)


server = AgentServer()


@server.rtc_session(agent_name="order-taker")
async def entrypoint(ctx: JobContext) -> None:
    warm_audio_resamplers()   # run before ANY session audio can start
    settings = load_settings()
    order = Order()                         # one fresh order per caller
    box = ToolBox(order, MenuSearch(settings))

    session = AgentSession(
        stt=mistralai.STT(
            model=STT_MODEL,
            language="en",
            context_bias=context_bias_words(),   # kottu, kiribath, coke...
        ),
        llm=mistralai.LLM(model=LLM_MODEL),
        tts=mistralai.TTS(model=TTS_MODEL, voice=TTS_VOICE, use_realtime=True),
    )
    agent = Agent(
        instructions=SYSTEM_PROMPT,
        tools=make_tools(box),
    )
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(
        instructions="Greet the caller warmly, one short sentence: welcome to "
                     "Spice Route Kitchen, what can I get you?")


if __name__ == "__main__":
    cli.run_app(server)
