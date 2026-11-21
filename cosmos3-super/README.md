# Cosmos3-Super on Google Cloud TPUs (TorchTPU)

- **Model:** [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) (27-shard physical-world generative video/simulation foundation model, `bf16`)
- **License:** NVIDIA Open Model License Agreement
- **Engines Audited:** `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` and `sglang-torchtpu` (`main`, `U=8` with tiled parallel VAE decode)

## Integration & Startup Audit Findings

1. **`vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`OmniTpuPlatform`, `U=4`):**
   - `vllm-omni` includes native `Cosmos3OmniDiffusersPipeline` support (`deploy/cosmos3_omni.yaml`). Default startup loads gated `nvidia/Cosmos-1.0-Guardrail` (`HTTP 401/403` without authorized access); passing **`--no-guardrails`** (or `--deploy-config .../vllm_omni/deploy/cosmos3_omni.yaml`, which sets `guardrails: false`) bypasses guardrail initialization cleanly.
   - During worker checkpoint loading, `vllm_omni/diffusion/models/cosmos3/audio_tokenizer/avae.py:82` calls `safetensors.torch.load_file(str(path), device=str(map_location))` with `device="tpu"`, which raises:
     `safetensors._safetensors_rust.SafetensorError: device tpu is invalid`
     (Rust `safetensors` requires loading to `"cpu"` before moving tensors to a TPU device).
2. **`sglang-torchtpu` (`main`, 4-chip TPU v7x, 8 TensorCores × 96 GB HBM/core, `Cosmos3OmniTransformer` 64 layers / ~108 GB `bf16`):**
   - **Local RAM-disk / NVMe staging:** Staging all 27 shards (`124 GB`) locally reduces server startup from `~29 min` (network object storage) to **`96.2 s`** (`/health 200 OK` with `--warmup-mode off`).
   - **Resident `bf16` (`U=8`, `--dit-cpu-offload false`):** Because `sglang` replicates transformer weights per core under sequence parallelism (`U=8`), loading the ~108 GB transformer + auxiliary modules exceeds `96 GB` HBM/core (`RESOURCE_EXHAUSTED: Attempting to allocate 500.00M. There are 387.67M free`).
   - **Tensor parallelism (`--tp-size 2` / `--tp-size 4`):** Explicitly unsupported on TPU diffusion today (`sglang/multimodal_gen/runtime/server_args/server_args.py:1677`: `ValueError: TPU diffusion has no tensor-parallel path yet; shard with --cfg-parallel-size / --sp-degree instead`).
   - **Online FP8 quantization (`--quantization fp8`):** Server boots in `75.2 s`, then request execution fails inside CUDA-Triton-only `sglang/kernels/ops/quantization/fp8_kernel.py:289` (`RuntimeError: 0 active drivers ([])`).
   - **Layerwise CPU offload (`--dit-layerwise-offload true --pin-cpu-memory false`):** Worker boot fails at `sglang/multimodal_gen/runtime/managers/memory_managers/layerwise_offload.py:1609` (`RuntimeError: Attempted to call variable.set_data(tensor), but variable and tensor have incompatible tensor type`).

## Reproduce

```bash
# Option A: SGLang on TorchTPU (4x TPU v7x / 8 cores, tiled parallel VAE decode, skip server warmup)
sglang serve \
  --model-path /path/to/nvidia/Cosmos3-Super \
  --host 0.0.0.0 --port 8000 \
  --num-gpus 8 --ulysses-degree 8 \
  --attention-backend tpu_splash_attn \
  --warmup-mode off \
  --vae-config.use-parallel-decode \
  --vae-config.parallel-decode-mode tiled

# Option B: vLLM-Omni on TorchTPU (--no-guardrails or deploy/cosmos3_omni.yaml)
vllm serve /path/to/nvidia/Cosmos3-Super \
  --omni --no-guardrails --served-model-name nvidia/Cosmos3-Super \
  --host 0.0.0.0 --port 8000 --ulysses-degree 4
```
