"""Black-box HTTP tests for the native client. Python is only the test server."""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest


BINARY = Path(sys.argv.pop(1)).resolve()


def event(value):
    return "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"


def choice(text="", finish=None, chat=True):
    return {"choices": [{"index": 0, "delta" if chat else "text":
                         {"content": text} if chat else text, "finish_reason": finish}]}


def counts(**changes):
    return {"prompt_tokens": 11, "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 3}, **changes}


def success(body, path, *, delay=0, usage=None, text="many words in one chunk"):
    chat = path.endswith("/chat/completions")
    usage = counts() if usage is None else usage
    if body.get("stream", False):
        return 200, "text/event-stream", [(delay, event(choice(text, "stop", chat)) +
            event({"choices": [], "usage": usage}) + event("[DONE]"))]
    value = {"choices": [{"index": 0, "finish_reason": "stop",
                          "message" if chat else "text":
                          {"role": "assistant", "content": text} if chat else text}], "usage": usage}
    return 200, "application/json", [(delay, json.dumps(value))]


@contextmanager
def server(respond=success):
    lock = threading.Lock()
    state = {"calls": [], "active": 0, "peak": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["calls"].append({"path": self.path, "body": body, "headers": dict(self.headers)})
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                status, content_type, frames = respond(body, self.path)
                encoded = [(delay, value.encode() if isinstance(value, str) else value) for delay, value in frames]
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(sum(len(value) for _, value in encoded)))
                self.end_headers()
                for delay, value in encoded:
                    time.sleep(delay)
                    self.wfile.write(value)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected for cancellation and rejected/truncated responses.
            finally:
                with lock:
                    state["active"] -= 1

        def log_message(self, *_):
            pass

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}/v1", state
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join()


def scenario(**changes):
    result = {"name": "test", "endpoint": "chat", "sessions": 1,
              "warmup_sessions": 0, "repetitions": 1, "concurrency": [1],
              "defaults": {"max_tokens": 32, "temperature": 0},
              "cases": [{"name": "plain", "body": {"messages": [{"role": "user", "content": "hello"}]}}]}
    result.update(changes)
    return {"version": 1, "scenarios": [result]}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def run_client(self, document, url, *, extra=(), expected=0, environment=None):
        self.counter += 1
        config = self.root / f"scenario-{self.counter}.json"
        config.write_text(json.dumps(document))
        output = self.root / f"result-{self.counter}"
        command = [str(BINARY), "--base-url", url, "--model", "test-model", "--scenarios", str(config),
                   "--output", str(output), *extra]
        if "--timeout-seconds" not in extra:
            command += ["--timeout-seconds", "3"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, env=environment)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text()) if (output / "summary.json").exists() else None
        records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()] if summary else []
        return result, output, summary, records

    def test_chat_and_text_streaming_and_buffered(self):
        for chat in [True, False]:
            for streaming in [True, False]:
                with self.subTest(chat=chat, streaming=streaming):
                    body = {"messages": [{"role": "user", "content": "hello"}]} if chat else {"prompt": "hello"}
                    body["stream"] = streaming
                    with server() as (url, state):
                        _, output, report, records = self.run_client(scenario(endpoint="chat" if chat else "text",
                            cases=[{"name": "plain", "body": body}]), url)
                    self.assertTrue(report["complete"])
                    stats = report["points"][0]["measured"]
                    self.assertEqual(stats["completed"], 1)
                    self.assertEqual(stats["known_completion_tokens"], 7)
                    self.assertGreater(stats["completion_tokens_per_second"], 0)
                    self.assertEqual(records[0]["text"], "many words in one chunk")
                    self.assertEqual(records[0]["content_events"], int(streaming))
                    self.assertEqual(records[0]["first_content_seconds"] is not None, streaming)
                    self.assertEqual(state["calls"][0]["path"], "/v1/chat/completions" if chat else "/v1/completions")
                    request = state["calls"][0]["body"]
                    self.assertEqual(request["model"], "test-model")
                    self.assertEqual(request["max_tokens"], 32)
                    self.assertEqual(request.get("stream_options"), {"include_usage": True} if streaming else None)
                    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"))
                    self.assertEqual(records[0]["request_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
                    manifest = json.loads((output / "manifest.json").read_text())
                    self.assertFalse(manifest["server_cache_reset"])
                    self.assertEqual(manifest["client_source_sha256"], hashlib.sha256(
                        Path(__file__).with_name("main.cc").read_bytes()).hexdigest())

    def test_sse_fragmentation_bom_comments_multiline_and_cr(self):
        body = ("\ufeff: keepalive\r\nevent: message\r\n\r\n" + event(choice()) +
                'data: {"choices": [{"index": 0,\r\ndata: "delta": {"content": "héllo"}, "finish_reason": null}]}\r\n\r\n' +
                event(choice(" world", "stop")) + event({"choices": [], "usage": counts()}) + event("[DONE]"))
        raw = body.encode()
        with server(lambda *_: (200, "text/event-stream", [(0, raw[i:i+1]) for i in range(len(raw))])) as (url, _):
            _, _, _, records = self.run_client(scenario(), url)
        self.assertEqual(records[0]["text"], "héllo world")
        self.assertEqual(records[0]["content_events"], 2)
        with server(lambda *_: (200, "text/event-stream", [(0, body.replace("\r\n", "\n").replace("\n", "\r"))])) as (url, _):
            self.run_client(scenario(), url)

    def test_first_content_ignores_empty_frames(self):
        frames = [(0, event(choice())), (.04, event(choice("first"))),
                  (.04, event(choice(" last", "stop")) + event({"usage": counts()}) + event("[DONE]"))]
        with server(lambda *_: (200, "text/event-stream", frames)) as (url, _):
            _, _, _, records = self.run_client(scenario(), url)
        self.assertGreaterEqual(records[0]["first_content_seconds"], .035)
        self.assertGreaterEqual(records[0]["max_content_gap_seconds"], .035)

    def test_warmup_matrix_repetitions_and_concurrency(self):
        document = scenario(sessions=4, warmup_sessions=1, repetitions=2, concurrency=[1, 2])
        with server(lambda body, path: success(body, path, delay=.015)) as (url, state):
            _, _, report, records = self.run_client(document, url)
        self.assertEqual(len(report["points"]), 4)
        self.assertEqual(len(state["calls"]), 20)
        self.assertEqual(sum(record["phase"] == "warmup" for record in records), 4)
        self.assertEqual(state["peak"], 2)
        for point in report["points"]:
            self.assertEqual(point["measured"]["completed"], 4)
            self.assertEqual(point["measured"]["known_completion_tokens"], 28)
            self.assertEqual(point["measured"]["peak_active_requests"], point["concurrency"])

    def test_scheduled_arrivals_include_queue_delay(self):
        with server(lambda body, path: success(body, path, delay=.04)) as (url, state):
            _, _, report, records = self.run_client(scenario(sessions=4, arrival_rates=[1000]), url)
        self.assertEqual(state["peak"], 1)
        self.assertGreater(report["points"][0]["measured"]["queue_seconds"]["max"], .05)
        for record in records:
            self.assertAlmostEqual(record["scheduled_seconds"], record["session"] / 1000)
            self.assertAlmostEqual(record["latency_seconds"], record["queue_seconds"] + record["http_seconds"])

    def test_arrivals_are_not_sent_early(self):
        with server() as (url, _):
            _, _, _, records = self.run_client(scenario(sessions=3, concurrency=[3], arrival_rates=[25]), url)
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertGreaterEqual(record["started_seconds"], record["session"] / 25)

    def test_growing_conversations_append_actual_answers(self):
        document = scenario(sessions=3, concurrency=[2], cases=[{
            "name": "conversation", "body": {"messages": [{"role": "user", "content": "hello"}]},
            "followups": [{"role": "user", "content": "continue"}, {"role": "user", "content": "summarize"}]}])
        with server() as (url, state):
            _, _, report, records = self.run_client(document, url)
        self.assertEqual(report["points"][0]["measured"]["completed"], 9)
        self.assertEqual(sorted(len(call["body"]["messages"]) for call in state["calls"]), [1]*3 + [3]*3 + [5]*3)
        for call in state["calls"]:
            messages = call["body"]["messages"]
            if len(messages) > 1:
                self.assertEqual(messages[1], {"role": "assistant", "content": "many words in one chunk"})
                self.assertEqual(messages[2]["content"], "continue")
        self.assertEqual(sorted((r["session"], r["turn"]) for r in records), [(s, t) for s in range(3) for t in range(3)])

    def test_mixed_cases_keep_structured_and_image_requests(self):
        cases = [
            {"name": "json", "body": {"messages": [{"role": "user", "content": "return JSON"}],
                                         "response_format": {"type": "json_object"}}},
            {"name": "image", "body": {"messages": [{"role": "user", "content": [
                {"type": "text", "text": "caption"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}]}]}},
        ]
        with server() as (url, state):
            _, _, _, records = self.run_client(scenario(cases=cases, sessions=4), url)
        self.assertEqual([r["case"] for r in records], ["json", "image", "json", "image"])
        self.assertEqual(state["calls"][0]["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(state["calls"][1]["body"]["messages"], cases[1]["body"]["messages"])

    def test_streaming_tool_arguments_are_reconstructed(self):
        def tool(delta, finish=None):
            return event({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        body = (tool({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "echo", "arguments": '{"text":'}}]}) +
                tool({"tool_calls": [{"index": 0, "function": {"arguments": '"hello"}'}}]}, "tool_calls") +
                event({"usage": counts()}) + event("[DONE]"))
        with server(lambda *_: (200, "text/event-stream", [(0, body)])) as (url, _):
            _, _, _, records = self.run_client(scenario(), url)
        self.assertIsNone(records[0]["first_content_seconds"])
        self.assertEqual(records[0]["message"]["tool_calls"][0]["function"], {"name": "echo", "arguments": '{"text":"hello"}'})

    def test_missing_usage_is_not_fabricated(self):
        body = event(choice("hello", "stop")) + event("[DONE]")
        with server(lambda *_: (200, "text/event-stream", [(0, body)])) as (url, _):
            _, _, report, records = self.run_client(scenario(), url)
        self.assertIsNone(records[0]["usage"])
        stats = report["points"][0]["measured"]
        self.assertEqual(stats["completed"], 1)
        self.assertEqual(stats["requests_with_usage"], 0)
        self.assertIsNone(stats["completion_tokens_per_second"])

    def test_invalid_usage_is_rejected(self):
        for value in [counts(completion_tokens=-1), counts(completion_tokens=True), counts(prompt_tokens=3.0),
                      counts(prompt_tokens_details={"cached_tokens": 12})]:
            with self.subTest(value=value):
                with server(lambda body, path: success(body, path, usage=value)) as (url, _):
                    _, _, report, records = self.run_client(scenario(), url, expected=1)
                self.assertEqual(report["unexpected_failures"], 1)
                self.assertEqual(records[0]["status"], "failed")

    def test_incomplete_and_invalid_streams_fail(self):
        for body in [event(choice("partial")), event(choice("done", "stop")) + "data: [DONE]\n",
                     event(choice("partial")) + event("[DONE]"), event({"error": {"message": "execution failed"}}),
                     event([]), event("{bad json"), event(choice("done", "stop")) + event(choice("late")) + event("[DONE]")]:
            with self.subTest(body=body):
                with server(lambda *_: (200, "text/event-stream", [(0, body)])) as (url, state):
                    _, _, report, records = self.run_client(scenario(), url, expected=1)
                self.assertEqual(len(state["calls"]), 1)
                self.assertEqual(report["unexpected_failures"], 1)
                self.assertTrue(records[0]["error"])

    def test_http_error_preserved_without_retry(self):
        with server(lambda *_: (503, "application/json", [(0, '{"error":"capacity exhausted"}')])) as (url, state):
            _, _, report, records = self.run_client(scenario(), url, expected=1)
        self.assertEqual(len(state["calls"]), 1)
        self.assertIn("HTTP 503", records[0]["error"])
        self.assertIn("capacity exhausted", records[0]["error"])
        self.assertEqual(report["points"][0]["measured"]["known_completion_tokens"], 0)

    def test_binary_error_body_does_not_abort_the_matrix(self):
        with server(lambda *_: (503, "text/plain", [(0, b"capacity \xff exhausted")])) as (url, state):
            _, _, report, records = self.run_client(scenario(sessions=2), url, expected=1)
        self.assertTrue(report["complete"])
        self.assertEqual(len(state["calls"]), 2)
        self.assertEqual(report["unexpected_failures"], 2)
        self.assertIn("capacity", records[0]["error"])

    def test_request_timeout_is_a_failure_not_an_intentional_cancellation(self):
        with server(lambda body, path: success(body, path, delay=.15)) as (url, _):
            _, _, report, records = self.run_client(scenario(), url, extra=["--timeout-seconds", ".025"], expected=1)
        self.assertEqual(records[0]["status"], "failed")
        self.assertEqual(report["points"][0]["measured"]["cancelled"], 0)

    def test_cancellation_and_response_limit(self):
        document = scenario(cases=[{"name": "cancel", "cancel_after_ms": 25,
                                   "body": {"messages": [{"role": "user", "content": "hello"}]}}])
        with server(lambda body, path: success(body, path, delay=.15)) as (url, _):
            _, _, report, records = self.run_client(document, url)
        self.assertEqual(records[0]["status"], "cancelled")
        self.assertEqual(report["points"][0]["measured"]["cancelled"], 1)
        self.assertLess(records[0]["http_seconds"], .13)
        with server() as (url, _):
            _, _, _, records = self.run_client(scenario(), url, extra=["--max-response-bytes", "8"], expected=1)
        self.assertIn("max-response-bytes", records[0]["error"])

    def test_authentication_and_direct_http_do_not_expose_key(self):
        secret = "test-secret-do-not-log"
        environment = dict(os.environ, TEST_BENCH_KEY=secret, HTTP_PROXY="http://127.0.0.1:1",
                           http_proxy="http://127.0.0.1:1", ALL_PROXY="http://127.0.0.1:1", NO_PROXY="")
        with server() as (url, state):
            result, output, _, _ = self.run_client(scenario(), url, extra=["--api-key-env", "TEST_BENCH_KEY"], environment=environment)
        self.assertEqual(state["calls"][0]["headers"]["Authorization"], "Bearer " + secret)
        self.assertNotIn(secret, result.stdout + result.stderr)
        for path in output.iterdir():
            self.assertNotIn(secret, path.read_text())
        keyfile = self.root / "key"
        keyfile.write_text(secret + "\n")
        with server() as (url, state):
            self.run_client(scenario(), url, extra=["--api-key-file", str(keyfile)])
        self.assertEqual(state["calls"][0]["headers"]["Authorization"], "Bearer " + secret)

    def test_plan_needs_no_server_and_preserves_output(self):
        result, output, report, _ = self.run_client(scenario(concurrency=[1, 2], arrival_rates=[0, 3], repetitions=2),
                                                   "http://127.0.0.1:1/v1", extra=["--plan"])
        self.assertEqual(len(json.loads(result.stdout)), 8)
        self.assertFalse(output.exists())
        self.assertIsNone(report)

    def test_invalid_scenarios_and_url_fail_before_output(self):
        invalid = [scenario(unknown=True), scenario(concurrency=[1, 1]), scenario(arrival_rates=[-1]),
                   scenario(sessions=True), scenario(defaults={"n": 2}), scenario(defaults={"stream_options": {"include_usage": False}}),
                   scenario(endpoint="text", cases=[{"name": "ids", "body": {"prompt": [1, 2]}}])]
        for document in invalid:
            with self.subTest(document=document):
                _, output, _, _ = self.run_client(document, "http://127.0.0.1:1/v1", extra=["--plan"], expected=2)
                self.assertFalse(output.exists())
        for url in ["file:///tmp/test", "http://user:secret@localhost/v1", "http://localhost/v1?key=secret"]:
            with self.subTest(url=url):
                _, output, _, _ = self.run_client(scenario(), url, extra=["--plan"], expected=2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
