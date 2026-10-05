# Apple Silicon (MPS / MLX): porting notes - plan only, nothing tested

Small model size alone does not make the nodes Mac-ready. What is CUDA-specific today:

- `euler_sample` (vendored) wraps the model in `torch.autocast("cuda", ...)`, enabled only for CUDA tensors. On MPS
  the model would run in its storage dtype without autocast: a different numeric path.
- The RoPE cache key includes the CUDA autocast state; tables are built in bfloat16 under autocast on CUDA. MPS would
  build them differently, so the result would not be bit-identical to CUDA.
- Initial noise comes from the CUDA generator (Philox). MPS and CPU generators give different numbers for the same
  seed. Drawing the noise on the CPU and moving it would make results reproducible across Macs, but not equal to CUDA.
- TF32 does not exist on MPS. FLAN-T5 runs in float32 (about 1.4 GB) and the DiT in bfloat16 (MPS supports bf16 on
  recent macOS).
- `F.scaled_dot_product_attention` has an MPS path; its kernels differ from CUDA.

A port would need its own reference (the official code run on MPS), its own tolerance, and real tests on the
hardware. None of that has been done.
