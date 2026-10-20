# Verification

## Sources (pinned)
| item | revision | sha256 |
| --- | --- | --- |
| OpenSenseNova/Looped-DiT (code, MIT) | `92a9c1914258361f78e03426588c7c215944b504` | vendored files: see looped_dit_comfy/vendor/UPSTREAM.md |
| sensenova/Looped-DiT-B16 `looped-dit-b16.pt` (MIT) | `7436a1994bc9d0d8d176525d0dad2be458fbe4df` | `2da674002833ed6cb7a7e9cceb3a680700ca251d06b6d1b117e083b8c6811eba` |
| google/flan-t5-large `model.safetensors` (Apache-2.0) | `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` | `a9dd06ce490f139af36e9eb77dd3758b4fd07a08a73d5a1abe5ff2591e2d388e` |

## Reference
- Official reference = the upstream functions `load_model`, `TextEncoder`, `euler_sample` and `to_pil`, called in the
  order of `pipeline.generate()` with `sample.py`'s TF32 setting, in a separate environment (torch 2.14.1+cu130,
  transformers 4.57.6 as upstream requires `< 5`).
- That scripted path was checked against the unmodified `python -m looped_dit.sample --loops 4 1 2` for the README
  example prompt ("a red cube on top of a blue sphere", seed 0): all three rows pixel-identical.
- Initial noise: the sampler's first `torch.randn` after `torch.manual_seed(seed)` was recorded with a pass-through
  wrapper (returns the same tensor) and equals the replica the node reports.

## What was compared (RTX 5090, Windows 11, batch 1, 512x512, 100 steps, CFG 6)
| level | result |
| --- | --- |
| token IDs and attention mask | identical (also 317 prompts on CPU vs transformers 4.57.6, `tests/fixtures/t5_tokens_transformers4.json`) |
| FLAN-T5 output (float32) | identical; transformers 5.18 (sdpa and eager) vs 4.57.6, 8 prompts |
| text cast to bfloat16 | identical |
| initial noise | identical (same hash for every loop depth of a sweep) |
| final float32 sample | identical, 27 of 27 |
| 8-bit image | identical, 27 of 27 (the PNG written by ComfyUI's Save Image) |

The 27 images are the README example prompt with seed 0 at loops 1/2/4, plus 4 prompts x seeds 7 and 2026 x loops 1/2/4
(docs/results/demo_manifest.json). All went through ComfyUI 0.38.0's HTTP queue (transformers 5.18.0, tokenizers 0.23.2).

### 0.1.1

0.1.1 disables two training-only functions of the vendored `config.py` (see `looped_dit_comfy/vendor/UPSTREAM.md`). The same 24
Loop Sweep images (4 prompts x 2 seeds x loops 1/2/4) were generated again on 2026-10-21 from a `git clone` of the 0.1.1 commit,
through ComfyUI 0.38.0's HTTP queue on the same RTX 5090: all 24 final float32 samples and all 24 8-bit images have the same SHA-256
as in docs/results/demo_manifest.json (0.1.0).

## ComfyUI behaviour checked through the real server (127.0.0.1, test instance)
- The 4 nodes register, and both API workflows pass ComfyUI's validation.
- Re-queueing the same graph after clearing ComfyUI's cache (`/free`) really executes again and gives the same pixels.
  Seed 0 -> 1 -> 0 and loops 4 -> 1 -> 4 also give identical results each time.
- Out-of-range input (`loops` 0) is rejected at validation (HTTP 400). A malformed loop list gives an execution error
  with a message.
- Cancel during a sweep ends with `execution_interrupted` and no saved outputs. The next run gives the reference pixels.
- Unload frees the GPU memory (3.84 to 2.05 GB in use, the rest being other processes), and the next run reloads and
  matches.
- ComfyUI's global RNG state and the TF32 flags are the same before and after a generation.

## Clean install and B/32
- A fresh copy from `git archive` of the tested commit was placed in a differently named `custom_nodes` folder. The
  models were installed with `tools/download_models.py` into an empty folder: real downloads at pinned revisions, all
  SHA-256 checks passed. The full server check above was then repeated, with the same results (bit-identical,
  re-execution, cancel, unload).
- B/32 (`looped-dit-b32.pt`, sha256 `175f89f2...`) with the README example prompt, seed 0, loops 1/2/4: the 8-bit image
  and float32 result are identical to the official code. Its 1/2/4 sweep shares one initial noise.
- Model switching B/16 -> B/32 -> B/16 -> B/32 in one server: B/16 still equals its reference, and B/32 repeats exactly.
- Memory, measured by the test harness on Windows: the ComfyUI process with the model loaded added at most 8.1 GiB to
  the system commit charge during these runs. The official reference process added 7.6-8.3 GiB. GPU memory in use peaked
  at about 3.9 GB, including about 1.4 GB used by other processes.

## CPU tests (CI)
`tests/` covers:
- the vendored files against their hashes and the upstream checkout;
- tokenization (317 prompts);
- generation on tiny random models, equal to the vendored official sampler;
- noise, RNG and TF32 side effects, cancel, argument bounds;
- checkpoint loading: `weights_only=True` refuses pickled code, strict keys, config bounds;
- Save Image round trip, the grid;
- registration and workflow validation in ComfyUI.

No model weights are used in CI.

## Not verified
- Other GPUs, drivers or torch builds; ComfyUI started with `--fast`, `--deterministic` or attention overrides.
- B/32 beyond the example prompt; CPU, MPS and MLX.
- Image quality beyond the author's own look at the demo images; no benchmark was run.
