"""Integration tests for the OpenAI server with the Kimi-K3 engine.

Validates that the server:
  - Starts with engine=kimik3 using the kimik3_tiny model
  - GET /health returns {"status": "ok"}
  - GET /v1/models returns a list containing "kimi-k3"
  - POST /v1/chat/completions returns choices[0].message.content
  - POST /v1/completions returns choices[0].text
  - Streaming chat completions produce SSE chunks

The tiny K3 model (kimik3_tiny/) has vocab=256 and no tokenizer.json, so the
server uses byte-level encoding (UTF-8 bytes as token IDs). Generated output is
deterministic (greedy by default) but the exact bytes depend on the random
weights — the tests only assert structural correctness, not content.

Run: python -m pytest tests/test_openai_server_kimik3.py -v
"""
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from openai_server import APIServer, K3Engine  # noqa: E402

K3_BIN = C_DIR / "kimik3"
K3_TINY = C_DIR / "kimik3_tiny"


class K3OpenAIServerTest(unittest.TestCase):
    """End-to-end tests: real K3 engine + tiny model behind the OpenAI server.

    Uses APIServer directly (like the existing HTTPTest in test_openai_server.py)
    rather than calling serve(), so there is no signal-handler/main-thread issue
    and the server can run cleanly in a background thread."""

    @classmethod
    def setUpClass(cls):
        if not K3_BIN.exists():
            raise unittest.SkipTest("kimik3 binary not built (run: make kimik3)")
        if not (K3_TINY / "config.json").exists():
            raise unittest.SkipTest("kimik3_tiny model not found")
        cls.engine = K3Engine(str(K3_BIN), str(K3_TINY), cap=8, max_tokens=16)
        cls.server = APIServer(("127.0.0.1", 0), cls.engine, "kimi-k3", None, 16,
                               kv_slots=1)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.scheduler.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.engine.close()

    def _request(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        data = json.dumps(body).encode() if body is not None else None
        req = Request(self.base + path, data=data, headers=headers)
        return urlopen(req, timeout=60)

    def test_health(self):
        with self._request("/health") as r:
            self.assertEqual(r.status, 200)
            body = json.load(r)
        self.assertEqual(body["status"], "ok")

    def test_models_lists_kimi_k3(self):
        with self._request("/v1/models") as r:
            self.assertEqual(r.status, 200)
            body = json.load(r)
        self.assertEqual(body["object"], "list")
        ids = [m["id"] for m in body["data"]]
        self.assertIn("kimi-k3", ids)

    def test_chat_completion_non_streaming(self):
        with self._request("/v1/chat/completions", {
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        }) as r:
            self.assertEqual(r.status, 200)
            body = json.load(r)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "kimi-k3")
        self.assertIn("choices", body)
        self.assertEqual(len(body["choices"]), 1)
        choice = body["choices"][0]
        self.assertIn("message", choice)
        self.assertIn("content", choice["message"])
        self.assertIsNone(choice["message"]["refusal"])
        self.assertIn("finish_reason", choice)
        self.assertIn("usage", body)
        self.assertIn("prompt_tokens", body["usage"])
        self.assertIn("completion_tokens", body["usage"])

    def test_chat_completion_streaming(self):
        req = Request(self.base + "/v1/chat/completions",
                      data=json.dumps({
                          "model": "kimi-k3",
                          "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 3,
                          "stream": True,
                      }).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers["Content-Type"], "text/event-stream")
            chunks = []
            saw_done = False
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    saw_done = True
                    break
                chunks.append(json.loads(data))
        self.assertTrue(saw_done, "stream did not end with [DONE]")
        self.assertTrue(len(chunks) >= 1, "no SSE chunks received")
        # First chunk should have the chat.completion.chunk object type.
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(chunks[0]["model"], "kimi-k3")

    def test_completions_endpoint(self):
        with self._request("/v1/completions", {
            "model": "kimi-k3",
            "prompt": "The quick brown fox",
            "max_tokens": 3,
        }) as r:
            self.assertEqual(r.status, 200)
            body = json.load(r)
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["model"], "kimi-k3")
        self.assertEqual(len(body["choices"]), 1)
        self.assertIn("text", body["choices"][0])

    def test_rejects_wrong_model(self):
        from urllib.error import HTTPError
        with self.assertRaises(HTTPError) as caught:
            self._request("/v1/chat/completions", {
                "model": "wrong-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            })
        self.assertEqual(caught.exception.code, 404)


class K3EngineUnitTest(unittest.TestCase):
    """Unit tests for K3Engine and render_chat_k3 without a live server."""

    @classmethod
    def setUpClass(cls):
        if not K3_BIN.exists():
            raise unittest.SkipTest("kimik3 binary not built")
        if not (K3_TINY / "config.json").exists():
            raise unittest.SkipTest("kimik3_tiny model not found")

    def test_render_chat_k3_basic(self):
        from openai_server import render_chat_k3
        prompt = render_chat_k3([
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ])
        self.assertIn("System: Be brief.", prompt)
        self.assertIn("User: Hello", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_render_chat_k3_rejects_empty(self):
        from openai_server import APIError, render_chat_k3
        with self.assertRaises(APIError):
            render_chat_k3([])

    def test_k3_engine_generate(self):
        eng = K3Engine(str(K3_BIN), str(K3_TINY), cap=8, max_tokens=4)
        chunks = []
        stats = eng.generate("hello", 3, 0.0, 1.0, chunks.append)
        eng.close()
        self.assertGreaterEqual(stats["completion_tokens"], 0)
        self.assertGreater(stats["prompt_tokens"], 0)
        # The engine should have called on_text at least once with decoded text.
        if stats["completion_tokens"] > 0:
            self.assertTrue(len(chunks) >= 1)

    def test_is_k3_engine_helper(self):
        from openai_server import is_k3_engine
        self.assertTrue(is_k3_engine("/path/to/kimik3"))
        self.assertTrue(is_k3_engine(K3_BIN))
        self.assertFalse(is_k3_engine("/path/to/colibri"))
        self.assertFalse(is_k3_engine(None))


if __name__ == "__main__":
    unittest.main()
