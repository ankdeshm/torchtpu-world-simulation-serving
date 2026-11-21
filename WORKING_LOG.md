# Working Log — World Simulation & Physical Reasoning on TorchTPU

## Goal
Bring up and benchmark world-simulation and physical-reasoning models on Google Cloud TPU v7x and TPU v5p using PyTorch / TorchTPU (`vllm-torchtpu`, `sglang-torchtpu`, and `torch.compile(backend="tpu")`).

## What We Tried & Found

### 1. TorchTPU Compilation & Device API Bring-Up (`V-JEPA2-ViT-g` & `Matrix-Game-4.0`)
- **Issue 1:** Calling `torch.tpu.set_device()` raised an internal-API `RuntimeError` on `torch_tpu 0.2.0`.
  - **Resolution:** Switched to standard PyTorch `torch.accelerator.set_device_index(0)` and `torch.accelerator.synchronize()`.
- **Issue 2:** Passing `torch.compile(..., backend="openxla")` fails without `torch_xla`.
  - **Resolution:** Switched to TorchTPU's `torch.compile(..., backend="tpu")`, which cut `V-JEPA2-ViT-g` `16f B=1` latency by `4.62×` on TPU v7x (`184.65 ms -> 40.00 ms`, `400.0 FPS`) and `6.62×` on TPU v5p (`287.65 ms -> 43.47 ms`), and lifted 5B `Matrix-Game-4.0` at full `KV=21` (`864×480`) from `20.1 FPS` eager to `68.56 FPS` (2-step) and `133.69 FPS` (1-step) on 1 TPU v7x core.

### 2. `Cosmos-Reason1-7B` Serving & Precision Ablation
- Benchmarked `vllm-torchtpu` (`v0.32.0` and `v0.33.0`) across `TP=2` (1 TPU v7x chip), `TP=4` (2 TPU v7x chips), and `TP=4` (4 TPU v5p chips) for both text (`1k/256`, `8k/512`) and multimodal (`mm_img_512_256`).
- A single TPU v7x chip (`TP=2`) reached `2,508.8 tok/s` (`c=64`), outperforming 4 TPU v5p chips (`1,910.9 tok/s`) by `1.31×`. Two TPU v7x chips (`TP=4`) reached `1,535.1 tok/s` (`c=16`, `TTFT 43.5 ms`) on multimodal image-text prompts.
- **Negative finding (`KV-FP8`):** Enabling `--kv-cache-dtype fp8` regressed `Cosmos-Reason1-7B` throughput by `-57%` on TPU v7x and `-73%` on TPU v5p compared to `BF16`. Default `BF16` KV cache is strongly recommended.
- **Open issue (`sglang-torchtpu`):** Scheduler initialization exited before serving `Cosmos-Reason1-7B` at `TP=1` and `TP=4` on TPU v5p. Re-audited at `TP=2` on TPU v7x under `sglang-torchtpu v0.2.0`: weight loading completed (`169.0 s`), then multimodal image placement failed at `sglang_tpu/mm_embed_tpu.py:139` (`RuntimeError: index_put_(): expected tensor to be on tpu, got cpu`).

### 3. `Cosmos3-Super` Bring-Up Findings
- On `vllm-omni 0.33.0rc1`, default orchestrator initialization fetches gated `nvidia/Cosmos-1.0-Guardrail` (`HTTP 401/403`); passing `--no-guardrails` (or `--deploy-config .../deploy/cosmos3_omni.yaml`) bypasses guardrail loading cleanly, after which `vllm_omni/diffusion/models/cosmos3/audio_tokenizer/avae.py:82` raises `safetensors._safetensors_rust.SafetensorError: device tpu is invalid` by passing `device="tpu"` directly to Rust `safetensors.torch.load_file`.
- On `sglang-torchtpu` (`main`, `U=8`), loading all 27 shards (`Cosmos3OmniTransformer`, `sp_size=8`) completed cleanly in ~29 min from network object storage, after which default server warmup (`Warmup requests: 0/1`) spent >30 min inside fp32 `F.conv3d` before `/health` reached `200 OK`; `--warmup-mode off` skips server warmup.
