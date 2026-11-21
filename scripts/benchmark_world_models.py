#!/usr/bin/env python3
"""PyTorch / TorchTPU World Simulation benchmark harness.

Benchmarks world-simulation and physical-reasoning models on Google Cloud TPU v7x and TPU v5p:
1. V-JEPA2-ViT-g (`facebook/vjepa2-vitg-fpc64-256`) eager vs `torch.compile(backend="tpu")`
2. Skywork/Matrix-Game-4.0 (5B CausalWanModel) eager vs `torch.compile(backend="tpu")`
3. Cosmos-Reason1-7B OpenAI-compatible text + multimodal chat serving (`vllm-torchtpu`)

Uses standard PyTorch `torch.accelerator` APIs with `torch_tpu`.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def make_test_rgb_b64(width: int = 512, height: int = 512) -> str:
    """Create a deterministic PNG data URL for multimodal VLM benchmarking."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(64, 128, 192))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def benchmark_vjepa2(model_path: str, frames: int, batch_size: int, compile_tpu: bool, iters: int) -> dict[str, Any]:
    import torch
    import torch_tpu  # noqa: F401
    from transformers import AutoModel

    torch.accelerator.set_device_index(0)
    device = torch.device("tpu:0")
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)
    load_s = time.perf_counter() - t0

    def run_step(x: torch.Tensor) -> torch.Tensor:
        out = model(pixel_values_videos=x)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    fn = torch.compile(run_step, backend="tpu") if compile_tpu else run_step
    x = torch.randn(batch_size, frames, 3, 256, 256, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        torch.accelerator.synchronize()
        tc0 = time.perf_counter()
        y = fn(x)
        torch.accelerator.synchronize()
        cold_ms = (time.perf_counter() - tc0) * 1000.0
        finite = bool(torch.isfinite(y.float()).all().item())

        lat_ms: list[float] = []
        for _ in range(iters):
            ts = time.perf_counter()
            y = fn(x)
            torch.accelerator.synchronize()
            lat_ms.append((time.perf_counter() - ts) * 1000.0)

    p50 = statistics.median(lat_ms)
    p95 = sorted(lat_ms)[min(len(lat_ms) - 1, int(0.95 * len(lat_ms)))]
    clips_per_s = batch_size / (p50 / 1000.0)
    return {
        "model": "facebook/vjepa2-vitg-fpc64-256",
        "mode": "torch.compile(backend='tpu')" if compile_tpu else "eager",
        "frames": frames,
        "batch_size": batch_size,
        "load_s": round(load_s, 2),
        "cold_ms": round(cold_ms, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "clips_per_sec": round(clips_per_s, 2),
        "frames_per_sec": round(clips_per_s * frames, 1),
        "finite": finite,
    }


def benchmark_cosmos_reason1(
    base_url: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    multimodal: bool,
) -> dict[str, Any]:
    """Benchmark Cosmos-Reason1-7B via OpenAI-compatible streaming chat completions."""
    img_url = make_test_rgb_b64() if multimodal else None
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    def one_req() -> tuple[float, float, int]:
        content: Any
        if multimodal:
            content = [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": "Analyze the physical layout and likely next motion in this scene in detail."},
            ]
        else:
            content = "Explain step-by-step how a rigid object slides down an inclined plane with friction and rebounds."
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        ttft = None
        out_toks = 0
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:].strip())
                choices = ev.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content") and ttft is None:
                    ttft = time.perf_counter() - t0
                if ev.get("usage"):
                    out_toks = int(ev["usage"].get("completion_tokens", out_toks))
        total = time.perf_counter() - t0
        return (ttft or total), total, max(out_toks, 1)

    t_wall0 = time.perf_counter()
    ttfts, totals, toks = [], [], []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in as_completed([pool.submit(one_req) for _ in range(num_requests)]):
            tf, tot, tk = fut.result()
            ttfts.append(tf * 1000.0)
            totals.append(tot)
            toks.append(tk)
    wall = time.perf_counter() - t_wall0
    return {
        "model": model,
        "multimodal": multimodal,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "output_tok_s": round(sum(toks) / wall, 2),
        "ttft_p50_ms": round(statistics.median(ttfts), 2),
        "tpot_p50_ms": round(statistics.median([(tot * 1000.0 - tf) / max(tk - 1, 1) for tf, tot, tk in zip(ttfts, totals, toks)]), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TorchTPU world simulation models.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_vj = sub.add_parser("vjepa2")
    p_vj.add_argument("--model-path", required=True)
    p_vj.add_argument("--frames", type=int, default=16)
    p_vj.add_argument("--batch-size", type=int, default=1)
    p_vj.add_argument("--compile", action="store_true")
    p_vj.add_argument("--iters", type=int, default=12)

    p_cr = sub.add_parser("cosmos-reason1")
    p_cr.add_argument("--base-url", default="http://127.0.0.1:8000")
    p_cr.add_argument("--model", default="nvidia/Cosmos-Reason1-7B")
    p_cr.add_argument("--concurrency", type=int, default=16)
    p_cr.add_argument("--num-requests", type=int, default=16)
    p_cr.add_argument("--max-tokens", type=int, default=256)
    p_cr.add_argument("--multimodal", action="store_true")

    args = parser.parse_args()
    if args.cmd == "vjepa2":
        res = benchmark_vjepa2(args.model_path, args.frames, args.batch_size, args.compile, args.iters)
    else:
        res = benchmark_cosmos_reason1(
            args.base_url, args.model, args.concurrency, args.num_requests, args.max_tokens, args.multimodal
        )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
