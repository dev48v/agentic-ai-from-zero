"""A tiny, dependency-free HTTP webhook receiver — the SECOND event source.

An event-driven agent fires on events, not on being called. A file/JSONL queue models the
"pull from a broker" source; this models the "an external system POSTs to a URL" source —
a Stripe payment, a GitHub push, a form submission. It is a stdlib `http.server` bound to
localhost that turns every inbound `POST /events` into an `Event` and `append()`s it to the
SAME `EventQueue` the worker drains. So a webhook delivery and a queued message converge on
one pipeline — the worker cannot tell (or care) which source an event came from.

Contract:
  POST /events   body: {"type": "new_order", "id": "...", "payload": {...}}
                 → 202 Accepted  {"status": "enqueued", "id": "<event id>"}
  GET  /health   → 200 {"status": "ok", "queue_depth": N}

`id` is optional; if omitted one is generated. Because the receiver only enqueues (it never
runs a handler), a webhook can never block on the model — delivery is fast and the reasoning
happens later on the worker, which is exactly the durability property you want.

Run standalone:   python 08-event-triggered/webhook.py --port 8808 --queue ./queue.jsonl
Then POST:        curl -X POST localhost:8808/events -d '{"type":"new_order","payload":{...}}'
`run.py` also drives this receiver in-process (a background thread) to prove the path end to
end without any external infra.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)

from queue import Event, EventQueue  # noqa: E402

logger = logging.getLogger("webhook")


def make_handler(event_queue: EventQueue, on_receive=None):
    """Build a request handler bound to a specific EventQueue (so multiple receivers /
    tests can each target their own queue file without global state)."""

    class WebhookHandler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                self._json(200, {"status": "ok", "queue_depth": event_queue.depth()})
            else:
                self._json(404, {"error": "not found", "hint": "POST /events or GET /health"})

        def do_POST(self):
            if self.path.split("?")[0] != "/events":
                self._json(404, {"error": "not found", "hint": "POST /events"})
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "body is not valid JSON"})
                return
            etype = data.get("type")
            if not etype:
                self._json(400, {"error": "missing 'type'"})
                return
            # Enqueue ONLY — never run a handler here, so a webhook can't block on the model.
            event = Event.new(type=etype, payload=data.get("payload", {}),
                              id=data.get("id"), source="webhook")
            event_queue.append(event)
            if on_receive:
                on_receive(event)
            self._json(202, {"status": "enqueued", "id": event.id, "type": event.type})

        def log_message(self, *args):  # silence the default stderr access log
            return

    return WebhookHandler


class WebhookReceiver:
    """A localhost webhook server managed as a context manager / start-stop pair, so a run
    (or a test) can spin one up on a background thread and tear it down deterministically."""

    def __init__(self, event_queue: EventQueue, host: str = "127.0.0.1",
                 port: int = 8808, on_receive=None) -> None:
        self.event_queue = event_queue
        self.host = host
        self._httpd = ThreadingHTTPServer((host, port), make_handler(event_queue, on_receive))
        self.port = self._httpd.server_address[1]     # actual port (0 → OS-assigned)
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "WebhookReceiver":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2)

    def __enter__(self) -> "WebhookReceiver":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Localhost webhook receiver → EventQueue")
    ap.add_argument("--port", type=int, default=8808)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--queue", default=os.path.join(_PROJECT_DIR, "queue.jsonl"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s")
    q = EventQueue(args.queue, reset=False)
    recv = WebhookReceiver(q, host=args.host, port=args.port,
                           on_receive=lambda e: logger.info("enqueued %s (%s)", e.id, e.type))
    recv.start()
    logger.info("webhook receiver on %s  →  queue %s", recv.url, args.queue)
    logger.info("POST /events  ·  GET /health  ·  Ctrl-C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        recv.stop()
        logger.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
