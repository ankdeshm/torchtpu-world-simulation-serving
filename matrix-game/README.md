# Skywork/Matrix-Game-4.0 on Google Cloud TPUs (TorchTPU)

- **Model:** [`Skywork/Matrix-Game-4.0`](https://huggingface.co/Skywork/Matrix-Game-4.0) (5.001B-parameter `CausalWanModel` streaming autoregressive world simulator, `864×480` resolution, 4 RGB frames per latent block, `bf16`)
- **License:** Apache-2.0
- **Execution Stack:** PyTorch `2.13.0` + `torch_tpu 0.2.0` (`eager` vs. `torch.compile(backend="tpu")`, 1 TPU core / chip)

## Measured Streaming DiT Results (`864×480`, 4 RGB Frames / Block, `finite=True`)

| TPU Hardware (1 Device) | Execution Mode | KV Window (Latents) | Denoise Steps | Block p50 | Block p95 | Streaming FPS (p50) | Compile Speedup vs. Eager |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 6 | 1 | **15.83 ms** | 16.30 ms | **252.75 FPS** | **2.68×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | 12 | 1 | **21.88 ms** | 22.28 ms | **182.83 FPS** | **2.83×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | **21 (full)** | **1** | **29.92 ms** | 30.24 ms | **133.69 FPS** | **3.33×** |
| **TPU v7x (1 core)** | **`torch.compile(backend="tpu")`** | **21 (full)** | **2** | **58.35 ms** | 58.77 ms | **68.56 FPS** | **3.41×** |
| TPU v7x (1 core) | Eager (`torch_tpu`) | 6 | 1 | 42.47 ms | 42.71 ms | 94.17 FPS | 1.00× |
| TPU v7x (1 core) | Eager (`torch_tpu`) | 12 | 1 | 61.96 ms | 62.03 ms | 64.56 FPS | 1.00× |
| TPU v7x (1 core) | Eager (`torch_tpu`) | 21 (full) | 1 | 99.77 ms | 100.12 ms | 40.09 FPS | 1.00× |
| TPU v7x (1 core) | Eager (`torch_tpu`) | 21 (full) | 2 | 199.06 ms | 199.13 ms | 20.09 FPS | 1.00× |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 6 | 1 | **24.27 ms** | 24.34 ms | **164.81 FPS** | **3.81×** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | 12 | 1 | **32.67 ms** | 32.75 ms | **122.43 FPS** | **3.02×** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | **21 (full)** | **1** | **50.72 ms** | 50.77 ms | **78.87 FPS** | **2.51×** |
| **TPU v5p (1 chip)** | **`torch.compile(backend="tpu")`** | **21 (full)** | **2** | **100.30 ms** | 100.50 ms | **39.88 FPS** | **2.42×** |
| TPU v5p (1 chip) | Eager (`torch_tpu`) | 21 (full) | 1 | 127.33 ms | 128.28 ms | 31.41 FPS | 1.00× |
| TPU v5p (1 chip) | Eager (`torch_tpu`) | 21 (full) | 2 | 242.40 ms | 244.44 ms | 16.50 FPS | 1.00× |

### Key Takeaways
- **Exceeds 60 FPS interactive real-time at full 2-step quality on a single TPU v7x TensorCore:** Compiling the 5.001B `CausalWanModel` with `torch.compile(backend="tpu")` reaches **`68.56 FPS` (`58.35 ms` per 4-frame block)** at full `KV=21` 2-step denoising and **`133.69 FPS` (`29.92 ms/block`)** at 1-step distilled denoising, with sub-millisecond p95 jitter (`30.24 ms`).
- **1.70×–1.72× TPU v7x speedup over TPU v5p at full context:** At full `KV=21`, a single TPU v7x core runs **`1.70×` faster** than a TPU v5p chip in 1-step mode (`133.69` vs `78.87 FPS`) and **`1.72×` faster** in 2-step mode (`68.56` vs `39.88 FPS`).
- **Scope note:** Measurements cover the streaming 5B `CausalWanModel` DiT forward pass with full causal KV cache window (`KV=6/12/21`); streaming VAE RGB decode and keyboard/camera input encoding run outside this isolated DiT block timer.
