"""A scripted OpenAI-compatible model server, for smoke tests without a model.

Each episode calls `describe`, then `backtest` and `submit` with a 20-period
reversal alpha. Which call comes next is read from the number of tool results
in the request, so concurrent episodes need no shared state. Serves
`POST /v1/chat/completions`, streaming or not, and `GET /v1/models`.

Usage:
    uv run python scripts/stub_model.py [--port 8765]

Then run the `stub` entry of `runs/models.toml` with `scripts/run_models.py`.
"""

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "stub"
ALPHA = (
    "class Reversal(Alpha):\n"
    "    def features(self, df):\n"
    "        return df.with_columns(\n"
    "            pl.col('ret').rolling_sum(20).over('asset').alias('ret20')\n"
    "        )\n"
    "    def predict(self, df):\n"
    "        return -pl.col('ret20')\n"
)
# The scripted calls, in order: tool name suffix and arguments.
SCRIPT: list[tuple[str, dict[str, Any]]] = [
    ("describe", {}),
    ("backtest", {"code": ALPHA}),
    ("submit", {"code": ALPHA, "claimed_sharpe": 0.5}),
]


def reply(request: dict[str, Any]) -> dict[str, Any]:
    """The assistant message answering a chat-completions request.

    Args:
        request: The request body.

    Returns:
        A message with the next scripted tool call, or plain text once the
        script is done.
    """
    done = sum(m.get("role") == "tool" for m in request.get("messages", []))
    if done >= len(SCRIPT):
        return {"role": "assistant", "content": "Submitted."}
    suffix, arguments = SCRIPT[done]
    names = [t["function"]["name"] for t in request.get("tools", [])]
    name = next((n for n in names if n.endswith(suffix)), suffix)
    call = {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return {"role": "assistant", "content": None, "tool_calls": [call]}


class Handler(BaseHTTPRequestHandler):
    """Answers chat completions with the script."""

    def do_GET(self) -> None:  # noqa: N802 - the name http.server calls.
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls.
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        message = reply(request)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        base = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": request.get("model", MODEL),
        }
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if not request.get("stream"):
            self._json(
                base
                | {
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
                    ],
                    "usage": usage,
                }
            )
            return
        delta = dict(message)
        for i, call in enumerate(delta.get("tool_calls") or []):
            call["index"] = i
        chunks = [
            {"index": 0, "delta": delta, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": finish},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for choice in chunks:
            chunk = base | {"object": "chat.completion.chunk", "choices": [choice]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        final = base | {
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": usage,
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())

    def _json(self, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    """Serves until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
