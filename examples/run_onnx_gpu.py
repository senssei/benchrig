#!/usr/bin/env python3
"""
Direct ONNX Runtime GenAI GPU Inference Runner.

Executes ONNX models (e.g., Phi-4-mini-instruct-cuda-gpu, Phi-3.5-mini) natively
on NVIDIA GeForce RTX GPUs via CUDAExecutionProvider, bypassing OS/CLI preview gates.
"""

import argparse
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def _bootstrap_cuda_env():
    """Ensure CUDA dynamic linker paths and library symlinks are configured before ONNX runtime initialization."""
    if os.environ.get("_ONNX_CUDA_BOOTSTRAPPED") == "1":
        return

    import glob
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    venv_nvidia = glob.glob(os.path.join(project_root, ".venv/lib/python*/site-packages/nvidia/*/lib"))
    ollama_cuda13 = "/usr/local/lib/ollama/cuda_v13"

    for d in venv_nvidia:
        if "cufft" in d:
            so11 = os.path.join(d, "libcufft.so.11")
            so12 = os.path.join(d, "libcufft.so.12")
            if os.path.exists(so11) and not os.path.exists(so12):
                try:
                    os.symlink("libcufft.so.11", so12)
                except Exception:
                    pass

    candidate_dirs = [ollama_cuda13] + venv_nvidia
    existing_dirs = [d for d in candidate_dirs if os.path.isdir(d)]

    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not all(d in cur_ld for d in existing_dirs):
        new_ld = ":".join(existing_dirs) + (":" + cur_ld if cur_ld else "")
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_ONNX_CUDA_BOOTSTRAPPED"] = "1"
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass


def main():
    _bootstrap_cuda_env()
    from core.onnx_client import OnnxGenAiClient
    parser = argparse.ArgumentParser(description="Direct ONNX Runtime GenAI GPU Inference Runner")
    parser.add_argument("--model", type=str, default="Phi-4-mini-instruct-cuda-gpu", help="Model name or directory in models/")
    parser.add_argument("--prompt", type=str, default="Write a python function to compute the nth Fibonacci number efficiently.", help="User prompt")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing models")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 ONNX Runtime GenAI - Direct GPU (CUDA) Inference")
    print("=" * 60)

    client = OnnxGenAiClient(models_dir=args.models_dir)

    print(f"Engine:       {client.engine_name}")
    print(f"OG Version:   {client.get_version()}")
    print(f"CUDA Active:  {client.is_cuda_available()}")

    installed = client.list_installed_models()
    print(f"\nDiscovered ONNX Models ({len(installed)}):")
    for m in installed:
        print(f"  • {m['name']} ({m['size'] / (1024**2):.1f} MB) -> {m['path']}")

    print(f"\nTarget Model: {args.model}")
    print(f"Prompt:       {args.prompt}")
    print("-" * 60)

    t0 = time.perf_counter()
    result = client.generate(
        model=args.model,
        prompt=args.prompt,
        system=args.system,
        options={
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )
    t1 = time.perf_counter()

    if not result.get("success"):
        print(f"❌ Inference failed: {result.get('error')}")
        sys.exit(1)

    print("\n📝 Generated Response:\n")
    print(result.get("response", "").strip())
    print("\n" + "=" * 60)
    print("📊 Inference Telemetry:")
    print(f"  • Time to First Token (TTFT): {result.get('ttft_sec', 0.0):.3f} s")
    print(f"  • Prefill Speed:              {result.get('prompt_tok_per_sec', 0.0):.1f} tok/s ({result.get('prompt_eval_count', 0)} prompt tokens)")
    print(f"  • Decode Speed:               {result.get('eval_tok_per_sec', 0.0):.1f} tok/s ({result.get('eval_count', 0)} completion tokens)")
    print(f"  • Total Inference Duration:   {result.get('total_time_sec', 0.0):.3f} s (Wall clock: {t1 - t0:.3f} s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
