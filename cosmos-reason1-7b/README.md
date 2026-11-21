# Cosmos-Reason1-7B on Google Cloud TPUs (TorchTPU)

- **Model:** [`nvidia/Cosmos-Reason1-7B`](https://huggingface.co/nvidia/Cosmos-Reason1-7B) (7B physical-AI vision-language reasoning model based on Qwen2.5-VL, `bf16`, `16K` context)
- **License:** NVIDIA Open Model License Agreement
- **Engines Tested:** `vllm-torchtpu` (`v0.32.0` and `v0.33.0`, `BF16` and `KV-FP8`) and `sglang-torchtpu` (`main`)

## Measured Results (`vllm-torchtpu`, `BF16` — Best Precision)

| Hardware & Topology | Engine | Scenario (`in/out`) | Concurrency | Output Throughput | TTFT p50 | TPOT p50 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TPU v7x (1 chip / `TP=2`)** | `vllm-torchtpu v0.32.0` | `text 1k/256` | 1 | **220.6 tok/s** | 38.3 ms | 3.75 ms |
| **TPU v7x (1 chip / `TP=2`)** | `vllm-torchtpu v0.32.0` | `text 1k/256` | 32 | 1,914.2 tok/s | 435.1 ms | 11.81 ms |
| **TPU v7x (1 chip / `TP=2`)** | `vllm-torchtpu v0.32.0` | `text 1k/256` | 64 | **2,508.8 tok/s** | 313.6 ms | 17.36 ms |
| **TPU v7x (1 chip / `TP=2`)** | `vllm-torchtpu v0.32.0` | `text 8k/512` | 16 | 1,119.4 tok/s | 115.6 ms | 10.90 ms |
| **TPU v7x (1 chip / `TP=2`)** | `vllm-torchtpu v0.32.0` | `mm_img 512×512 + 256` | 16 | 1,155.2 tok/s | 183.1 ms | 8.37 ms |
| **TPU v7x (2 chips / `TP=4`)** | `vllm-torchtpu v0.32.0` | `text 1k/256` | 64 | 2,345.6 tok/s | 446.8 ms | 18.07 ms |
| **TPU v7x (2 chips / `TP=4`)** | `vllm-torchtpu v0.32.0` | `text 8k/512` | 1 | **272.7 tok/s** | 91.6 ms | **3.14 ms** |
| **TPU v7x (2 chips / `TP=4`)** | `vllm-torchtpu v0.32.0` | `text 8k/512` | 16 | **1,294.1 tok/s** | 132.3 ms | 9.41 ms |
| **TPU v7x (2 chips / `TP=4`)** | `vllm-torchtpu v0.32.0` | `mm_img 512×512 + 256` | 4 | 884.8 tok/s | **23.8 ms** | 3.53 ms |
| **TPU v7x (2 chips / `TP=4`)** | `vllm-torchtpu v0.32.0` | `mm_img 512×512 + 256` | 16 | **1,535.1 tok/s** | **43.5 ms** | **6.70 ms** |
| **TPU v7x (2 chips / `TP=4`)** | `vllm-torchtpu v0.33.0` | `text 8k/512` | 1 | **275.5 tok/s** | 105.5 ms | **3.10 ms** |
| **TPU v7x (2 chips / `TP=4`)** | `vllm-torchtpu v0.33.0` | `mm_img 512×512 + 256` | 4 | **1,010.0 tok/s** | **23.8 ms** | **3.25 ms** |
| **TPU v7x (2 chips / `TP=4`)** | `vllm-torchtpu v0.33.0` | `mm_img 512×512 + 256` | 16 | 1,399.2 tok/s | 107.0 ms | 6.69 ms |
| TPU v5p (4 chips / `TP=4`) | `vllm-torchtpu v0.32.0` | `text 1k/256` | 1 | 216.7 tok/s | 52.6 ms | 3.84 ms |
| TPU v5p (4 chips / `TP=4`) | `vllm-torchtpu v0.32.0` | `text 1k/256` | 64 | 1,910.9 tok/s | 672.4 ms | 20.09 ms |
| TPU v5p (4 chips / `TP=4`) | `vllm-torchtpu v0.32.0` | `text 8k/512` | 1 | 239.3 tok/s | 117.1 ms | 3.58 ms |
| TPU v5p (4 chips / `TP=4`) | `vllm-torchtpu v0.32.0` | `mm_img 512×512 + 256` | 16 | 923.3 tok/s | 66.6 ms | 9.22 ms |

### Key Takeaways & Negative Findings
- **A single TPU v7x chip (`TP=2`) outperforms 4-chip TPU v5p (`TP=4`) by 1.31×:** On `text 1k/256 c=64`, 1 TPU v7x chip delivers **`2,508.8 tok/s`** vs. `1,910.9 tok/s` on 4 TPU v5p chips (`5.25×` higher throughput per chip).
- **Strong multimodal serving on both `v0.32.0` and `v0.33.0`:** On 2 TPU v7x chips (`TP=4`), `mm_img_512_256` reaches **`1,535.1 tok/s`** at `c=16` (`43.5 ms` TTFT) on `v0.32.0` and **`1,010.0 tok/s`** at `c=4` (`23.8 ms` TTFT) on `v0.33.0` — **`1.66×` faster** than 4-chip TPU v5p (`923.3 tok/s`).
- **Avoid `--kv-cache-dtype fp8` on `Cosmos-Reason1-7B`:** Enabling `KV-FP8` severely regressed throughput on both TPU v7x (`1,002.6` vs `2,345.6 tok/s`, `-57%`) and TPU v5p (`512.6` vs `1,910.9 tok/s`, `-73%`). Keep `BF16` KV cache for this 7B dense VLM.
- **`sglang-torchtpu` compatibility:** Tested on TPU v5p (`TP=1` and `TP=4`); scheduler initialization exited before serving `Cosmos-Reason1-7B`. Use `vllm-torchtpu` (`v0.32.0` or `v0.33.0`) for production serving.

## Reproduce

```bash
# Single TPU v7x chip (TP=2) for highest throughput per chip, or TP=4 for lowest multimodal latency
vllm serve /path/to/nvidia/Cosmos-Reason1-7B \
  --served-model-name nvidia/Cosmos-Reason1-7B \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --trust-remote-code

python3 ../scripts/benchmark_world_models.py cosmos-reason1 \
  --model nvidia/Cosmos-Reason1-7B --concurrency 16 --num-requests 16 --multimodal
```
