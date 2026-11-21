# Appendix — World Simulation Ablations, Negative Results & Engine Diagnostics

This appendix records full quantization ablations, batch/context sweeps, compilation timings, and engine compatibility boundaries for world-simulation models on TPU v7x and TPU v5p.

---

## 1. `Cosmos-Reason1-7B`: `BF16` vs. `KV-FP8` Precision Ablation (Negative Result for `KV-FP8`)

While `--kv-cache-dtype fp8` can benefit larger sparse MoE models at low concurrency, enabling `KV-FP8` on dense 7B `Cosmos-Reason1-7B` caused substantial throughput regressions across both TPU v7x and TPU v5p:

| Hardware & Topology | Precision | `text 1k/256 c=1` | `text 1k/256 c=64` | `text 8k/512 c=16` | `mm_img 512×512 c=16` | Change vs. `BF16` (`c=64`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x (`TP=2`, 1 chip)** | **`BF16`** | **220.6 tok/s** | **2,508.8 tok/s** | 1,119.4 tok/s | 1,155.2 tok/s | Baseline |
| **TPU v7x (`TP=4`, 2 chips)** | **`BF16`** | 206.4 tok/s | **2,345.6 tok/s** | **1,294.1 tok/s** | **1,535.1 tok/s** | Baseline |
| TPU v7x (`TP=4`, 2 chips) | `KV-FP8` | 62.1 tok/s | 1,002.6 tok/s | 95.4 tok/s | 136.3 tok/s | **-57.3%** |
| **TPU v5p (`TP=4`, 4 chips)** | **`BF16`** | **216.7 tok/s** | **1,910.9 tok/s** | **1,184.9 tok/s** | **923.3 tok/s** | Baseline |
| TPU v5p (`TP=4`, 4 chips) | `KV-FP8` | 31.9 tok/s | 512.6 tok/s | 53.5 tok/s | 36.6 tok/s | **-73.2%** |

**Recommendation:** Always serve `Cosmos-Reason1-7B` with default `BF16` KV cache on TorchTPU.

---

## 2. `V-JEPA2-ViT-g` (1.0B): Eager vs. `torch.compile(backend="tpu")` Batch Scaling

One-time XLA compile latency for `torch.compile(backend="tpu")` was `51.6–63.7 s` on TPU v7x and `65.5–81.4 s` on TPU v5p for `B=1..32` at `16` frames and `B=1..2` at `64` frames.

| Hardware (1 Device) | Mode | `16f B=1` | `16f B=4` | `16f B=8` | `16f B=16` | `16f B=32` | `64f B=1` | `64f B=8` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x** | **`tpu` compile** | **40.00 ms** (`400.0 FPS`) | 153.53 ms (`416.9 FPS`) | 286.90 ms (`446.1 FPS`) | 563.65 ms (`454.2 FPS`) | 1,121.67 ms (**`456.5 FPS`**) | **257.09 ms** (`248.9 FPS`) | 2,086.30 ms (`245.4 FPS`) |
| TPU v7x | Eager | 184.65 ms (`86.7 FPS`) | 298.08 ms (`214.7 FPS`) | 369.65 ms (`346.3 FPS`) | 644.24 ms (`397.4 FPS`) | 1,191.57 ms (`429.7 FPS`) | 371.91 ms (`172.1 FPS`) | 2,087.11 ms (`245.3 FPS`) |
| **TPU v5p** | **`tpu` compile** | **43.47 ms** (`368.1 FPS`) | 167.11 ms (`383.0 FPS`) | 315.33 ms (`405.9 FPS`) | 607.38 ms (`421.5 FPS`) | 1,214.24 ms (**`421.7 FPS`**) | **265.77 ms** (`240.8 FPS`) | 2,328.13 ms (`219.9 FPS`) |
| TPU v5p | Eager | 287.65 ms (`55.6 FPS`) | 358.33 ms (`178.6 FPS`) | 479.16 ms (`267.1 FPS`) | 789.18 ms (`324.4 FPS`) | 1,503.52 ms (`340.5 FPS`) | 448.35 ms (`142.7 FPS`) | 2,328.80 ms (`219.9 FPS`) |

Note that at `64f B=4` and `64f B=8`, both eager and compiled runs converge (`~1.34 s` / `~2.09 s` on v7x) because large 64-frame sequences are already compute-bound on the MXU, whereas `16f B=1` is dispatch-bound in eager mode and benefits dramatically (`4.62×–6.62×`) from `torch.compile(backend="tpu")`.

---

## 3. `Skywork/Matrix-Game-4.0` (5.001B): Cold Compile & Causal KV Window Scaling

One-time compilation took `11.0–12.2 s` per static KV window shape on TPU v7x (`864×480`, 4 RGB frames per latent block, `finite=True` across all 16 configurations):

- **TPU v7x vs. TPU v5p (`torch.compile(backend="tpu")`):**
  - `KV=6` (1-step): `15.83 ms` (`252.75 FPS`) vs. `24.27 ms` (`164.81 FPS`) -> **1.53× faster on v7x**
  - `KV=12` (1-step): `21.88 ms` (`182.83 FPS`) vs. `32.67 ms` (`122.43 FPS`) -> **1.49× faster on v7x**
  - `KV=21` (1-step): `29.92 ms` (`133.69 FPS`) vs. `50.72 ms` (`78.87 FPS`) -> **1.70× faster on v7x**
  - `KV=21` (2-step): `58.35 ms` (`68.56 FPS`) vs. `100.30 ms` (`39.88 FPS`) -> **1.72× faster on v7x**

---

## 4. TorchTPU API & Engine Compatibility Notes

1. **Use `torch.compile(..., backend="tpu")` (not `backend="openxla"`):** Passing `backend="openxla"` attempts to import `torch_xla`. Native TorchTPU registers the `tpu` TorchDynamo backend directly.
2. **Use `torch.accelerator.set_device_index(0)` and `torch.accelerator.synchronize()`:** Direct calls to internal `torch.tpu.set_device()` raise a `RuntimeError` on `torch_tpu 0.2.0`.
3. **`Cosmos-Reason1-7B` on `sglang-torchtpu`:** Tested at `TP=1` and `TP=4` on TPU v5p (scheduler startup exited before serving) and at `TP=2` on TPU v7x under `sglang-torchtpu v0.2.0` (loaded weights in `169.0 s`, then raised `RuntimeError: index_put_(): expected tensor to be on tpu, got cpu` inside `sglang_tpu/mm_embed_tpu.py:139` because `_HostRowBuffer.__init__` allocates `torch.zeros(...)` without `device="cpu"`). Use `vllm-torchtpu` (`v0.32.0` or `v0.33.0`).
4. **`Cosmos3-Super` gated guardrail dependency:** `vllm-omni 0.33.0rc1` pulls `nvidia/Cosmos-1.0-Guardrail` on startup (HTTP 401 without gated HF token acceptance). On `sglang-torchtpu`, cold object-storage shard loading across 27 shards exceeded an 1800 s timeout at 25/27 shards; pre-stage weights on local disk or raise startup timeout `>=3600 s`.
