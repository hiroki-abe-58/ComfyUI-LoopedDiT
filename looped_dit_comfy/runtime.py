"""Loading and sampling Looped-DiT inside ComfyUI.

Follows looped_dit/pipeline.py and looped_dit/sample.py of the official repository (commit 92a9c19):
- load_checkpoint() = pipeline.load_model(): torch.load(weights_only=True), TrainConfig.from_dict(ckpt["config"]),
  LoopedMMDiT(**cfg.model_kwargs()), strict load of ckpt["ema"], bfloat16, eval;
- generate() = pipeline.generate() with the TF32 setting of sample.py: FLAN-T5 encoder in float32 -> text cast to the
  model dtype -> torch seeded right before the vendored euler_sample() -> pipeline.to_pil()'s pixel conversion.
Differences that do not change the numbers: the seed is applied inside torch.random.fork_rng so ComfyUI's global RNG
state is restored afterwards; TF32 is switched on only for the call and restored; a pass-through wrapper checks
ComfyUI's cancel flag and reports progress before every model forward.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .tokenization import T5Tokenization
from .vendor.config import TrainConfig
from .vendor.diffusion import euler_sample
from .vendor.model import LoopedMMDiT

UPSTREAM_COMMIT = "92a9c1914258361f78e03426588c7c215944b504"
OFFICIAL_CHECKPOINTS = {
    "2da674002833ed6cb7a7e9cceb3a680700ca251d06b6d1b117e083b8c6811eba": "sensenova/Looped-DiT-B16@7436a19 looped-dit-b16.pt",
    "175f89f29c024732763dc4e36fd76aa581d94c96dfea06a4bf41d9c5552e4ecd": "sensenova/Looped-DiT-B32@83a623e looped-dit-b32.pt",
}
OFFICIAL_T5 = {"a9dd06ce490f139af36e9eb77dd3758b4fd07a08a73d5a1abe5ff2591e2d388e": "google/flan-t5-large@0613663 model.safetensors"}
T5_FILES = ("config.json", "tokenizer.json", "model.safetensors")
MAX_LOOPS, MAX_STEPS = 16, 1000

_LOCK = threading.RLock()
_SLOT: Loaded | None = None
_HASHES: dict[tuple, str] = {}


@dataclass
class Loaded:
    key: tuple
    model: torch.nn.Module
    cfg: TrainConfig
    t5: torch.nn.Module
    tok: T5Tokenization
    device: torch.device
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha256(path: str | os.PathLike) -> str:
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def _check_limits(cfg: TrainConfig) -> None:
    """Refuse configs far outside the released models (a hostile file could otherwise request a huge model)."""
    pre, core, post = (int(v) for v in cfg.loop_split)
    checks = {
        "image_size": 16 <= cfg.image_size <= 1024 and cfg.image_size % cfg.patch_size == 0,
        "patch_size": 2 <= cfg.patch_size <= 64,
        "hidden_size": 16 <= cfg.hidden_size <= 4096,
        "heads": 1 <= cfg.num_heads <= 64 and 8 <= cfg.head_dim <= 256,
        "mlp_ratio": 0 < cfg.mlp_ratio <= 8,
        "pca_channels": 1 <= cfg.pca_channels <= 4096,
        "text_dim": 8 <= cfg.text_dim <= 8192,
        "text_preamble_depth": 0 <= cfg.text_preamble_depth <= 8,
        "loop_split": 1 <= min(pre, core, post) and pre + core + post <= 64,
        "num_loops": 1 <= cfg.num_loops <= MAX_LOOPS,
        "prompt_length": 8 <= cfg.prompt_length <= 512,
        "noise_scale": 0 < cfg.noise_scale <= 10,
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"checkpoint config outside the supported range: {bad}")


@contextlib.contextmanager
def official_precision():
    """looped_dit/sample.py enables TF32 'as in training and evaluation'; enable it for this call only."""
    with _LOCK:
        old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def _rng_devices(device: torch.device) -> list[int]:
    return [device.index] if device.type == "cuda" else []


def _seed(device: torch.device, seed: int) -> None:
    # sample.py calls torch.manual_seed(seed); the sampler's noise comes from this device's generator only
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


class _Watched(torch.nn.Module):
    """Pass-through: run a callback (cancel check, progress) before each forward of the wrapped model."""

    def __init__(self, inner: torch.nn.Module, before_forward):
        super().__init__()
        self.inner = inner
        self._before = before_forward

    def forward(self, *args, **kwargs):
        if self._before is not None:
            self._before()
        return self.inner(*args, **kwargs)


def official_uint8(x: torch.Tensor) -> torch.Tensor:
    """looped_dit.pipeline.to_pil()'s conversion; returns [B, H, W, 3] uint8."""
    images = (x.float().clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
    return images.permute(0, 2, 3, 1).contiguous()


def uint8_to_comfy_image(u8: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE: float32 [B, H, W, C] in [0, 1]; exactly the 8-bit image (SaveImage writes the same bytes back)."""
    return u8.cpu().float() / 255.0


# --------------------------------------------------------------------------- loading


def load_checkpoint(path: str | os.PathLike, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)  # no unsafe-pickle fallback
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("config"), dict) or not isinstance(ckpt.get("ema"), dict):
        raise ValueError("not a Looped-DiT checkpoint (expected a dict with 'config' and 'ema')")
    cfg = TrainConfig.from_dict(ckpt["config"])  # unknown keys raise
    _check_limits(cfg)
    with torch.random.fork_rng(devices=[]):  # module init draws from the CPU RNG; leave ComfyUI's state as it was
        model = LoopedMMDiT(**cfg.model_kwargs())
    model.load_state_dict(ckpt["ema"])  # strict
    del ckpt
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False), cfg


def load_text_encoder(t5_dir: str | os.PathLike, device: torch.device) -> torch.nn.Module:
    from transformers import T5EncoderModel

    model = T5EncoderModel.from_pretrained(str(t5_dir), local_files_only=True, use_safetensors=True, attn_implementation="eager")
    if next(model.parameters()).dtype != torch.float32:  # the official TextEncoder runs FLAN-T5 in float32
        model = model.float()
    return model.to(device).eval().requires_grad_(False)


def _load(key: tuple, verify_hashes: bool) -> Loaded:
    ckpt_path, t5_dir, device_name = key
    device = torch.device(device_name)
    info: dict = {"checkpoint_file": Path(ckpt_path).name, "text_encoder_dir": Path(t5_dir).name, "device": device_name,
                  "upstream_commit": UPSTREAM_COMMIT}
    if verify_hashes:
        t = time.perf_counter()
        ck = file_sha256(ckpt_path)
        t5 = file_sha256(Path(t5_dir) / "model.safetensors")
        info.update(checkpoint_sha256=ck, checkpoint_release=OFFICIAL_CHECKPOINTS.get(ck, "not an official release file"),
                    text_encoder_sha256=t5, text_encoder_release=OFFICIAL_T5.get(t5, "not the official FLAN-T5-Large file"),
                    hash_s=round(time.perf_counter() - t, 2))
    t = time.perf_counter()
    model, cfg = load_checkpoint(ckpt_path, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    info["load_model_s"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    t5m = load_text_encoder(t5_dir, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    info["load_text_encoder_s"] = round(time.perf_counter() - t, 3)
    d_model = getattr(t5m.config, "d_model", None)
    if d_model != cfg.text_dim:
        raise ValueError(f"text encoder width {d_model} does not match the checkpoint's text_dim {cfg.text_dim}")
    tok = T5Tokenization(Path(t5_dir) / "tokenizer.json", cfg.prompt_length)
    info["config"] = {k: getattr(cfg, k) for k in ("image_size", "patch_size", "loop_split", "num_loops", "use_xsa", "use_attn_gate",
                                                    "noise_scale", "prompt_length")}
    return Loaded(key, model, cfg, t5m, tok, device, info)


def acquire(checkpoint: str, text_encoder: str, device: str = "cuda:0", verify_hashes: bool = True) -> Loaded:
    """The single resident model: reuse it if it matches, otherwise free it and load the requested one."""
    global _SLOT
    key = (os.path.realpath(checkpoint), os.path.realpath(text_encoder), device)
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        unload()
        _SLOT = _load(key, verify_hashes)
        return _SLOT


def unload() -> bool:
    global _SLOT
    with _LOCK:
        had = _SLOT is not None
        _SLOT = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return had


def resident() -> dict | None:
    with _LOCK:
        return None if _SLOT is None else dict(_SLOT.info)


# --------------------------------------------------------------------------- sampling


def _check_args(cfg: TrainConfig, seed: int, loops: int, steps: int, cfg_scale: float) -> None:
    if not 0 <= int(seed) < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    if not 1 <= int(loops) <= MAX_LOOPS:
        raise ValueError(f"loops must be in [1, {MAX_LOOPS}]")
    if not 1 <= int(steps) <= MAX_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_STEPS}]")
    if not 1.0 <= float(cfg_scale) <= 30.0:
        raise ValueError("cfg must be in [1, 30]")


def generate(loaded: Loaded, prompt: str, seed: int, loops: int, steps: int = 100, cfg_scale: float = 6.0,
             before_forward=None) -> dict:
    """One image. Returns {"image": IMAGE, "uint8": [1,H,W,3], "report": {...}}."""
    cfg, dev = loaded.cfg, loaded.device
    _check_args(cfg, seed, loops, steps, cfg_scale)
    ids, mask = loaded.tok([prompt])
    ids, mask = ids.to(dev), mask.to(dev)
    dtype = next(loaded.model.parameters()).dtype
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad(), official_precision():
        t = time.perf_counter()
        text32 = loaded.t5(input_ids=ids, attention_mask=mask).last_hidden_state
        if cuda:
            torch.cuda.synchronize(dev)
        enc_s = time.perf_counter() - t
        text = text32.to(dtype)
        with torch.random.fork_rng(devices=_rng_devices(dev)):
            _seed(dev, int(seed))
            noise = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=dev, dtype=text.dtype) * cfg.noise_scale
            noise_sha = tensor_sha(noise)  # the draw euler_sample makes next (checked against the official run)
            del noise
            _seed(dev, int(seed))
            t = time.perf_counter()
            x = euler_sample(_Watched(loaded.model, before_forward), text, mask, cfg.image_size, int(steps), float(cfg_scale),
                             cfg.noise_scale, int(loops))
            if cuda:
                torch.cuda.synchronize(dev)
            den_s = time.perf_counter() - t
    t = time.perf_counter()
    u8 = official_uint8(x)
    image = uint8_to_comfy_image(u8)
    post_s = time.perf_counter() - t
    pre, core, post = (int(v) for v in cfg.loop_split)
    passes = 2 if float(cfg_scale) != 1.0 else 1
    report = {
        "prompt": prompt, "seed": int(seed), "loops": int(loops), "steps": int(steps), "cfg": float(cfg_scale),
        "image_size": cfg.image_size, "tokens": int(mask.sum()), "prompt_length": cfg.prompt_length,
        "model_forwards": int(steps) * passes, "passes_per_step": passes,
        "double_stream_blocks_per_forward": pre + core * int(loops) + post, "text_blocks_per_forward": cfg.text_preamble_depth,
        "double_stream_block_runs": int(steps) * passes * (pre + core * int(loops) + post),
        "input_ids_sha": tensor_sha(ids), "mask_sha": tensor_sha(mask), "text_fp32_sha": tensor_sha(text32),
        "text_sha": tensor_sha(text), "noise_sha": noise_sha, "final_float_sha": tensor_sha(x), "uint8_sha": tensor_sha(u8[0]),
        "encode_s": round(enc_s, 3), "denoise_s": round(den_s, 3), "postprocess_s": round(post_s, 4),
        "dtype": str(dtype), "tf32_during_call": True,
    }
    if cuda:
        report["cuda_max_allocated_mib"] = torch.cuda.max_memory_allocated(dev) // 2**20
        report["cuda_max_reserved_mib"] = torch.cuda.max_memory_reserved(dev) // 2**20
    return {"image": image, "uint8": u8.cpu(), "report": report}


def parse_loops(text: str) -> list[int]:
    items = [s.strip() for s in str(text).replace(";", ",").split(",") if s.strip()]
    if not items:
        raise ValueError("give at least one loop depth, e.g. '1, 2, 4'")
    if len(items) > 8:
        raise ValueError("at most 8 loop depths per sweep")
    out = []
    for s in items:
        if not s.isdigit():
            raise ValueError(f"not a loop depth: {s!r}")
        v = int(s)
        if not 1 <= v <= MAX_LOOPS:
            raise ValueError(f"loop depth {v} outside [1, {MAX_LOOPS}]")
        out.append(v)
    return out


def labeled_grid(u8_images: list[torch.Tensor], labels: list[str]) -> torch.Tensor:
    """Side-by-side images with a label bar above each; returns uint8 [1, H, W, 3]."""
    from PIL import Image, ImageDraw, ImageFont

    h, w = u8_images[0].shape[1], u8_images[0].shape[2]
    bar, gap = 40, 8
    canvas = Image.new("RGB", (len(u8_images) * w + (len(u8_images) - 1) * gap, h + bar), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for i, (img, label) in enumerate(zip(u8_images, labels)):
        x0 = i * (w + gap)
        canvas.paste(Image.fromarray(img[0].numpy()), (x0, bar))
        draw.text((x0 + 10, 9), label, fill=(240, 240, 240), font=font)
    import numpy as np

    return torch.from_numpy(np.array(canvas))[None]
