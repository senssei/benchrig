"""Tests for the generic-cpu-on-CUDA warning helper (plan.md Phase 2, item 2.1)."""

import unittest


class WarningForProviderTests(unittest.TestCase):
    def test_generic_cpu_on_nvidia_host_emits_warning(self):
        # Import inside the test so a missing module reads as "not found" / red.
        from benchrig.core.runtimes import warning_for_provider

        msg = warning_for_provider(provider="generic-cpu", host_gpu_type="nvidia")
        self.assertIsNotNone(msg, "expected a warning for generic-cpu on a CUDA host")
        # The warning must name the provider and the GPU host so the user knows what to change.
        self.assertIn("generic-cpu", msg)
        self.assertIn("cuda", msg.lower())

    def test_generic_cpu_on_apple_silicon_emits_no_warning(self):
        # On Apple Silicon, generic-cpu IS the normal CPU execution provider, not the slow path.
        from benchrig.core.runtimes import warning_for_provider

        self.assertIsNone(warning_for_provider(provider="generic-cpu", host_gpu_type="apple_silicon"))

    def test_cuda_provider_on_nvidia_emits_no_warning(self):
        from benchrig.core.runtimes import warning_for_provider

        self.assertIsNone(warning_for_provider(provider="cuda", host_gpu_type="nvidia"))

    def test_no_provider_emits_no_warning(self):
        from benchrig.core.runtimes import warning_for_provider

        self.assertIsNone(warning_for_provider(provider=None, host_gpu_type="nvidia"))
        self.assertIsNone(warning_for_provider(provider="", host_gpu_type="nvidia"))

    def test_warning_text_is_stable(self):
        """The warning text is part of the spec (spec.md §3); protect against accidental rewording."""
        from benchrig.core.runtimes import warning_for_provider

        msg = warning_for_provider(provider="generic-cpu", host_gpu_type="nvidia")
        self.assertIn("2", msg)
        self.assertIn("22", msg)
        self.assertIn("tok/s", msg)


if __name__ == "__main__":
    unittest.main()
