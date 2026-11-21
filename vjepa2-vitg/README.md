# V-JEPA2-ViT-g on Google Cloud TPUs (TorchTPU)

- **Model:** [`facebook/vjepa2-vitg-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256) (1.0B-parameter ViT-giant self-supervised world-model video encoder, `256×256` resolution, `bf16`)
- **License:** MIT
- **Execution Stack:** PyTorch `2.13.0` + `torch_tpu 0.2.0` (`eager` vs. `torch.compile(backend="tpu")`, 1 TPU core / chip)

## Measured Results (`finite=True` Verified on Every Run)

| TPU Hardware (1 Device) | Execution Mode | Frames | Batch | Latency p50 | Latency p95 | Throughput (clips/s) | Throughput (FPS) | Compile Speedup vs. Eager |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 16 | 1 | **40.00 ms** | 40.54 ms | **25.00** | **400.0 FPS** | **4.62×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 16 | 4 | 153.53 ms | 154.42 ms | 26.05 | 416.9 FPS | 1.94× |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 16 | 8 | 286.90 ms | 287.32 ms | 27.88 | 446.1 FPS | 1.29× |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 16 | 16 | 563.65 ms | 564.31 ms | 28.39 | 454.2 FPS | 1.14× |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 16 | 32 | 1,121.67 ms | 1,123.77 ms | **28.53** | **456.5 FPS** | 1.06× |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 64 | 1 | **257.09 ms** | 257.42 ms | **3.89** | **248.9 FPS** | **1.45×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 64 | 2 | 531.02 ms | 531.77 ms | 3.77 | 241.0 FPS | 1.42× |
| TPU v7x (1 core) | Eager (`torch_tpu`) | 16 | 1 | 184.65 ms | 190.25 ms | 5.42 | 86.7 FPS | 1.00× |
| TPU v7x (1 core) | Eager (`torch_tpu`) | 16 | 32 | 1,191.57 ms | 1,193.10 ms | 26.86 | 429.7 FPS | 1.00× |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 16 | 1 | **43.47 ms** | 45.20 ms | **23.00** | **368.1 FPS** | **6.62×** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 16 | 8 | 315.33 ms | 315.81 ms | 25.37 | 405.9 FPS | 1.52× |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 16 | 32 | 1,214.24 ms | 1,214.75 ms | **26.35** | **421.7 FPS** | 1.24× |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 64 | 1 | **265.77 ms** | 266.10 ms | **3.76** | **240.8 FPS** | **1.69×** |
| TPU v5p (1 chip) | Eager (`torch_tpu`) | 16 | 1 | 287.65 ms | 340.25 ms | 3.48 | 55.6 FPS | 1.00× |
| TPU v5p (1 chip) | Eager (`torch_tpu`) | 16 | 32 | 1,503.52 ms | 1,504.78 ms | 21.28 | 340.5 FPS | 1.00× |

### Key Takeaways
- **`torch.compile(backend="tpu")` removes host-dispatch overhead completely at low batch sizes:** At `16f B=1`, compiling cuts latency **4.62× on TPU v7x** (`184.65 ms -> 40.00 ms`, **`400.0 FPS`**) and **6.62× on TPU v5p** (`287.65 ms -> 43.47 ms`, **`368.1 FPS`**), with near-zero tail jitter (`p95 = 40.54 ms` on v7x).
- **Near-peak MXU utilization even at `B=1`:** With `torch.compile(backend="tpu")`, a single `16f B=1` clip already reaches **87.6% of peak `B=32` throughput** (`400.0` vs `456.5 FPS` on TPU v7x), enabling real-time robotics/world-model perception without waiting to form large batches.
- **Note on `backend="tpu"` vs. `backend="openxla"`:** Native TorchTPU compilation uses `torch.compile(..., backend="tpu")`; passing `backend="openxla"` requires `torch_xla` and is not used with `torch_tpu`.

## Reproduce

```bash
python3 ../scripts/benchmark_world_models.py vjepa2 \
  --model-path /path/to/facebook/vjepa2-vitg-fpc64-256 \
  --frames 16 --batch-size 1 --compile --iters 12
```
