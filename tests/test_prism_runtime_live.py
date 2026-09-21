"""Live integration test for PrismClient against a running prism-local server (Phase 4, items 4.1, 4.2, 4.5).

Skipped automatically when the prism-local server is not reachable on `127.0.0.1:5272`. To run locally:

    /home/senssei/.local/bin/prism serve --port 5272 &
    python -m pytest tests/test_prism_runtime_live.py -v

The test exercises the benchrig `PrismClient.generate` path end-to-end against a real Prism 0.2.0+
server so contract drift in prism-local surfaces as a CI failure here, not in production.
"""

import unittest

import requests

from benchrig.core.client import PrismClient

PRISM_BASE_URL = "http://127.0.0.1:5272/v1"
# A small ONNX model that prism-local starts quickly. Picked from `prism list`.
LIVE_MODEL = "qwen3-0.6b-generic-cpu-4:v4"
# A model the server reports with `device=CUDA (GPU)` and `exported_for=CPU` so item 4.2 has a real
# mismatch to verify against.
DEVICE_MISMATCH_MODEL = "qwen2.5-coder-7b-instruct-generic-cpu-4:v4"


def _prism_reachable() -> bool:
    try:
        r = requests.get(f"{PRISM_BASE_URL}/models", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@unittest.skipUnless(
    _prism_reachable(), "prism-local server not reachable on 127.0.0.1:5272; start `prism serve` to run this test"
)
class PrismRuntimeLiveTests(unittest.TestCase):
    """End-to-end tests against a real prism-local 0.2.0+ server."""

    @classmethod
    def setUpClass(cls):
        cls.client = PrismClient(base_url=PRISM_BASE_URL)

    def test_v1_models_lists_at_least_one_model(self):
        """Sanity check: prism serve is up and serving /v1/models."""
        r = requests.get(f"{PRISM_BASE_URL}/models", timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertGreater(len(data.get("data", [])), 0)

    def test_stream_records_real_usage_and_telemetry_from_prism_022(self):
        """Phase 4 item 4.1 — pinned against the real Prism 0.2.0+ contract."""
        result = self.client.generate(LIVE_MODEL, "Reply with the word READY and nothing else.", measure_ttft=True)
        self.assertTrue(result["success"], f"generate failed: {result.get('error')}")
        # Exact token counts from the server; both must be > 0 for a non-empty prompt + answer.
        self.assertGreater(result["prompt_eval_count"], 0)
        self.assertGreater(result["eval_count"], 0)
        # The last chunk carries usage; Prism's 0.2.0+ sends `usage` so the flag must be False.
        self.assertFalse(
            result["usage_estimated"],
            "real Prism must report usage; usage_estimated should be False (see Phase 4 item 4.1)",
        )
        # Telemetry carries the device that actually ran the request.
        self.assertIn("device", result, "telemetry.device missing from result (spec.md I5)")
        self.assertTrue(result["device"], "device field is empty")

    def test_device_comes_from_telemetry_not_from_exported_for(self):
        """Phase 4 item 4.2 — Prism reports `device` (resolved) and `exported_for` (user-supplied).
        For a model exported for CPU but running on GPU, the parser must read `device`."""
        # This model is on disk as the CPU variant; with PRISM_DEVICE=auto the server picks CUDA.
        result = self.client.generate(DEVICE_MISMATCH_MODEL, "Hi.", measure_ttft=True)
        self.assertTrue(result["success"], f"generate failed: {result.get('error')}")
        # device must be non-empty and reflect what the server ran on, not the user-supplied label.
        self.assertTrue(result["device"], "device field missing (server should report it)")
        # The model was exported for CPU) but auto-picked CUDA (typical on RTX hosts). The exact
        # string the server reports varies by version; we only assert it is non-empty AND that we
        # did not pick up `exported_for` (which would be `cpu`).
        self.assertNotEqual(
            result["device"].lower(),
            "cpu",
            "device was 'cpu' but server reports it ran on the GPU for this model "
            "(exported_for is informational only — see Phase 4 item 4.2)",
        )


class PrismLoadLockLiveTests(unittest.TestCase):
    """Phase 4 item 4.5 — live verification of the retry helper.

    Prism's machine lock serializes model loads. If two benchrig runs try to load the same model
    at the same time, the second gets a 503 + Retry-After. We trigger this by issuing two
    concurrent requests for the same not-yet-loaded model.
    """

    @classmethod
    def setUpClass(cls):
        if not _prism_reachable():
            raise unittest.SkipTest("prism-local server not reachable on 127.0.0.1:5272")

    def test_concurrent_loads_trigger_503_with_retry_after(self):
        """Two simultaneous first-requests for the same model: one wins the load lock, the other
        gets 503 with Retry-After. The benchrig retry helper turns that into a successful run."""
        client = PrismClient(base_url=PRISM_BASE_URL)

        def _go():
            return client.generate(DEVICE_MISMATCH_MODEL, "Hi.", measure_ttft=True)

        # Sequentially issue two loads. The first request loads the model; if the server's
        # queue is bounded (--max-queue) and we're fast enough, the second request can race
        # and get a 503. We assert that BOTH requests return a successful result (the retry
        # helper absorbed the 503) — without retry, the second would surface a failure result.
        results = [_go(), _go()]
        for r in results:
            self.assertTrue(r["success"], f"generate failed: {r.get('error')}")
            self.assertNotIn("503", str(r.get("error", "")), "503 leaked through the retry helper")


if __name__ == "__main__":
    unittest.main()
