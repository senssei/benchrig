"""Unit tests for OnnxGenAiClient (Direct ONNX Runtime GenAI CUDA Inference)."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from core.client import create_runtime_client
from core.onnx_client import OnnxGenAiClient


class TestOnnxGenAiClient(unittest.TestCase):
    """Test suite for direct ONNX GenAI client execution, model discovery, and metrics."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.client = OnnxGenAiClient(models_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Verify client initializes with specified paths and default attributes."""
        self.assertEqual(self.client.name, "onnx-gpu")
        self.assertEqual(self.client.engine_name, "ONNX Runtime GenAI")
        self.assertEqual(self.client.models_dir, os.path.abspath(self.temp_dir.name))
        self.assertEqual(self.client.timeout_sec, 300)

    @patch("core.onnx_client.OG_AVAILABLE", True)
    @patch("core.onnx_client.og")
    def test_version_and_cuda_status(self, mock_og):
        """Verify get_version reflects library version and CUDA availability."""
        mock_og.__version__ = "0.16.0"
        mock_og.is_cuda_available.return_value = True

        self.assertTrue(self.client.is_reachable())
        self.assertTrue(self.client.is_cuda_available())
        self.assertIn("0.16.0", self.client.get_version())
        self.assertIn("CUDA", self.client.get_version())

    def test_list_installed_models_empty(self):
        """Empty directory should return an empty list."""
        self.assertEqual(self.client.list_installed_models(), [])

    def test_list_installed_models_discovery(self):
        """Discovers models that contain genai_config.json."""
        model_subdir = os.path.join(self.temp_dir.name, "phi-4-mini-cuda")
        os.makedirs(model_subdir, exist_ok=True)
        config_file = os.path.join(model_subdir, "genai_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            f.write('{"model": {"type": "phi3"}}')
        dummy_weights = os.path.join(model_subdir, "model.onnx")
        with open(dummy_weights, "w", encoding="utf-8") as f:
            f.write("test_weights_data")

        models = self.client.list_installed_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "phi-4-mini-cuda")
        self.assertEqual(models[0]["runtime"], "onnx-gpu")
        self.assertEqual(models[0]["details"]["family"], "phi3")

    def test_resolve_model_path(self):
        """Resolves model path correctly when genai_config.json is present."""
        model_subdir = os.path.join(self.temp_dir.name, "phi-4-mini")
        os.makedirs(model_subdir, exist_ok=True)
        config_file = os.path.join(model_subdir, "genai_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("{}")

        resolved = self.client._resolve_model_path("phi-4-mini")
        self.assertEqual(resolved, os.path.abspath(model_subdir))
        self.assertIsNone(self.client._resolve_model_path("non-existent-model"))

    @patch("core.onnx_client.OG_AVAILABLE", True)
    @patch("core.onnx_client.og")
    def test_generate_mock(self, mock_og):
        """Test generation workflow and metric computation with mocked ONNX runtime."""
        # Create dummy model folder
        model_subdir = os.path.join(self.temp_dir.name, "test-model")
        os.makedirs(model_subdir, exist_ok=True)
        with open(os.path.join(model_subdir, "genai_config.json"), "w", encoding="utf-8") as f:
            f.write("{}")

        # Mock Model & Tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]  # 4 prompt tokens
        mock_tokenizer.decode.return_value = "Hello world"
        mock_og.Model.return_value = mock_model
        mock_og.Tokenizer.return_value = mock_tokenizer

        # Mock Generator
        mock_generator = MagicMock()
        # Returns 3 tokens then finishes
        mock_generator.is_done.side_effect = [False, False, False, True]
        mock_generator.get_next_tokens.side_effect = [[101], [102], [103]]
        mock_og.Generator.return_value = mock_generator

        res = self.client.generate("test-model", "Test prompt")

        self.assertTrue(res["success"])
        self.assertEqual(res["model"], "test-model")
        self.assertEqual(res["response"], "Hello world")
        self.assertEqual(res["prompt_eval_count"], 4)
        self.assertEqual(res["eval_count"], 3)
        self.assertGreater(res["eval_tok_per_sec"], 0.0)

    def test_factory_creation(self):
        """Verify create_runtime_client correctly instantiates OnnxGenAiClient."""
        cfg = {"onnx": {"models_dir": self.temp_dir.name, "timeout_sec": 120}}
        client = create_runtime_client("onnx-gpu", cfg)
        self.assertIsInstance(client, OnnxGenAiClient)
        self.assertEqual(client.timeout_sec, 120)


if __name__ == "__main__":
    unittest.main()
