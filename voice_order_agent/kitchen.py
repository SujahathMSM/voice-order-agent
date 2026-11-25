"""Step 10b: the kitchen screen. Run: uv run python -m voice_order_agent.kitchen

Deliberately boring: stdlib http server, the page polls /orders.json every 3s.
No frameworks, no websockets — v1 is polling; upgrade only if it feels bad live.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .store import OrderStore

store = OrderStore()   # reads only; writes happen in the agent process

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Kitchen - Spice Route</title>
<style>
  body { font-family: system-ui, sans-serif; background: #1e2430; color: #eee;
         margin: 0; padding: 24px; }
  h1 { font-size: 20px; color: #ffd166; }
  #orders { display: flex; flex-wrap: wrap; gap: 16px; }
  .card { background: #2a3242; border-radius: 12px; padding: 16px 20px;
          min-width: 240px; box-shadow: 0 2px 8px rgba(0,0,0,.35); }
  .card h2 { margin: 0 0 8px; color: #8be58b; font-size: 16px; }
  .card .meta { color: #ffd166; margin-bottom: 8px; }
  .card li { margin: 2px 0; }
  .empty { color: #888; }
  .stamp { color: #777; font-size: 12px; margin-left: 12px; }
</style></head><body>
<h1>Spice Route Kitchen &mdash; incoming orders <span class="stamp" id="stamp"></span></h1>
<div id="orders"><p class="empty">waiting for confirmed orders...</p></div>
<script>
async function refresh() {
  const res = await fetch('/orders.json');
  const orders = await res.json();
  const box = document.getElementById('orders');
  box.innerHTML = orders.length ? '' : '<p class="empty">waiting for confirmed orders...</p>';
  for (const o of orders) {
    const items = o.items.map(i => `<li>${i.quantity} &times; ${i.name}`
        + (i.options ? ` <small>(${i.options})</small>` : '') + `</li>`).join('');
    box.insertAdjacentHTML('beforeend',
      `<div class="card"><h2>Order #${o.id}</h2>
       <div class="meta">pickup ${o.pickup_time} &bull; ${o.total} LKR</div>
       <ul>${items}</ul>
       <div class="stamp">confirmed ${o.confirmed_at}</div></div>`);
  }
  document.getElementById('stamp').textContent =
      'updated ' + new Date().toLocaleTimeString();
}
refresh();
setInterval(refresh, 3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/orders.json":
            body = json.dumps(store.recent_orders()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):   # keep the console quiet
        pass


def main() -> None:
    addr = ("127.0.0.1", 8765)
    print(f"Kitchen screen: http://{addr[0]}:{addr[1]}  (Ctrl+C to stop)")
    HTTPServer(addr, Handler).serve_forever()


if __name__ == "__main__":
    main()
