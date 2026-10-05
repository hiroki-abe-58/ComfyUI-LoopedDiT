"""Runtime behaviour on CPU with tiny random models (no weights): equality with the vendored official sampler, RNG and
TF32 side effects, loop sweeps, cancellation, checkpoint loading safety, image conversion."""

import pickle

import numpy as np
import pytest
import torch

from looped_dit_comfy import runtime
from looped_dit_comfy.vendor.config import TrainConfig
from looped_dit_comfy.vendor.diffusion import euler_sample
from looped_dit_comfy.vendor.model import LoopedMMDiT

TINY = {"image_size": 32, "patch_size": 8, "hidden_size": 64, "num_heads": 2, "head_dim": 32, "mlp_ratio": 2.0, "pca_channels": 16,
        "text_dim": 16, "text_preamble_depth": 1, "loop_split": [1, 2, 1], "num_loops": 3, "prompt_length": 8, "noise_scale": 2.0}


class FakeTok:
    def __call__(self, prompts):
        ids = torch.zeros(len(prompts), 8, dtype=torch.long)
        mask = torch.zeros(len(prompts), 8, dtype=torch.long)
        for i, p in enumerate(prompts):
            n = min(8, len(p.split()) + 1)
            ids[i, :n] = torch.arange(1, n + 1)
            mask[i, :n] = 1
        return ids, mask


class FakeT5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(3)
        self.emb = torch.nn.Parameter(torch.randn(64, 16, generator=g))

    def forward(self, input_ids, attention_mask):
        class Out:
            pass

        o = Out()
        o.last_hidden_state = self.emb[input_ids] * attention_mask[..., None]
        return o


def tiny_loaded(dtype=torch.float32):
    cfg = TrainConfig.from_dict({**TINY, "deep_supervision": True})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LoopedMMDiT(**cfg.model_kwargs())
        torch.nn.init.normal_(model.final.weight, std=0.02)
    model = model.to(dtype).eval()
    return runtime.Loaded(("tiny",), model, cfg, FakeT5().eval(), FakeTok(), torch.device("cpu"), {})


def official_path(loaded, prompt, seed, loops, steps, cfg_scale):
    """What looped_dit/sample.py does: torch.manual_seed(seed) then pipeline.generate() pieces."""
    ids, mask = loaded.tok([prompt])
    text = loaded.t5(input_ids=ids, attention_mask=mask).last_hidden_state.to(next(loaded.model.parameters()).dtype)
    torch.manual_seed(seed)
    return euler_sample(loaded.model, text, mask, loaded.cfg.image_size, steps, cfg_scale, loaded.cfg.noise_scale, loops)


@pytest.mark.parametrize("loops", [1, 2, 3, 5])
def test_generate_equals_official_sampler(loops):
    loaded = tiny_loaded()
    out = runtime.generate(loaded, "a red cube", seed=11, loops=loops, steps=4, cfg_scale=6.0)
    ref = official_path(loaded, "a red cube", 11, loops, 4, 6.0)
    assert out["report"]["final_float_sha"] == runtime.tensor_sha(ref)
    assert torch.equal(out["uint8"], runtime.official_uint8(ref))
    assert out["image"].shape == (1, 32, 32, 3) and out["image"].dtype == torch.float32
    assert 0.0 <= float(out["image"].min()) and float(out["image"].max()) <= 1.0
    assert out["report"]["model_forwards"] == 8 and out["report"]["double_stream_blocks_per_forward"] == 1 + 2 * loops + 1


def test_noise_report_is_the_sampler_draw():
    loaded = tiny_loaded()
    out = runtime.generate(loaded, "x", seed=5, loops=2, steps=2, cfg_scale=1.0)
    torch.manual_seed(5)
    noise = torch.randn(1, 3, 32, 32) * 2.0
    assert out["report"]["noise_sha"] == runtime.tensor_sha(noise)
    assert out["report"]["model_forwards"] == 2 and out["report"]["passes_per_step"] == 1


def test_global_rng_and_tf32_are_left_unchanged():
    loaded = tiny_loaded()
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    runtime.generate(loaded, "x", seed=9, loops=2, steps=2, cfg_scale=6.0)
    assert torch.equal(state, torch.get_rng_state())
    assert tf32 == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_same_noise_across_loop_depths_and_reuse_is_stateless():
    loaded = tiny_loaded()
    a4 = runtime.generate(loaded, "x y", 3, 4, 3, 6.0)["report"]
    a1 = runtime.generate(loaded, "x y", 3, 1, 3, 6.0)["report"]
    b4 = runtime.generate(loaded, "x y", 3, 4, 3, 6.0)["report"]
    assert a4["noise_sha"] == a1["noise_sha"] == b4["noise_sha"]
    assert a4["final_float_sha"] == b4["final_float_sha"] != a1["final_float_sha"]


def test_cancel_stops_and_restores_state():
    loaded = tiny_loaded()
    calls = {"n": 0}

    def before():
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("cancelled")

    state = torch.get_rng_state().clone()
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(KeyboardInterrupt):
        runtime.generate(loaded, "x", 1, 2, 10, 6.0, before_forward=before)
    assert calls["n"] == 3
    assert torch.equal(state, torch.get_rng_state())
    assert tf32 == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    ok = runtime.generate(loaded, "x", 1, 2, 2, 6.0)  # still usable
    assert ok["image"].shape == (1, 32, 32, 3)


@pytest.mark.parametrize("bad", [dict(seed=-1), dict(loops=0), dict(loops=17), dict(steps=0), dict(steps=1001), dict(cfg_scale=0.5)])
def test_argument_bounds(bad):
    loaded = tiny_loaded()
    args = {"seed": 0, "loops": 1, "steps": 1, "cfg_scale": 6.0, **bad}
    with pytest.raises(ValueError):
        runtime.generate(loaded, "x", **args)


def test_parse_loops():
    assert runtime.parse_loops("1, 2,4") == [1, 2, 4]
    for bad in ("", "0", "1, x", "1;2;3;4;5;6;7;8;9", "17", "-1"):
        with pytest.raises(ValueError):
            runtime.parse_loops(bad)


def test_comfy_image_round_trips_through_saveimage():
    u8 = torch.arange(256, dtype=torch.uint8).repeat(3, 1).T.reshape(1, 16, 16, 3)
    img = runtime.uint8_to_comfy_image(u8)
    saved = np.clip(255.0 * img.numpy(), 0, 255).astype(np.uint8)  # ComfyUI SaveImage
    assert np.array_equal(saved, u8.numpy())


def test_official_uint8_formula(upstream_dir):
    import sys

    sys.path.insert(0, upstream_dir)
    try:
        from looped_dit.pipeline import to_pil
    finally:
        sys.path.remove(upstream_dir)
    x = torch.linspace(-1.2, 1.2, 3 * 16 * 16).reshape(1, 3, 16, 16)
    assert np.array_equal(np.array(to_pil(x)[0]), runtime.official_uint8(x)[0].numpy())


# --------------------------------------------------------------------------- checkpoint loading


def _save(obj, path):
    torch.save(obj, path)
    return path


def tiny_ckpt(tmp_path, **cfg_over):
    cfg = TrainConfig.from_dict({**TINY, **cfg_over})
    model = LoopedMMDiT(**cfg.model_kwargs())
    return _save({"config": cfg.to_dict(), "ema": model.state_dict()}, tmp_path / "tiny.pt"), model


def test_load_checkpoint_strict_and_rng_neutral(tmp_path):
    path, model = tiny_ckpt(tmp_path)
    state = torch.get_rng_state().clone()
    loaded, cfg = runtime.load_checkpoint(path, torch.device("cpu"), torch.float32)
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), model.state_dict().values()))
    assert cfg.loop_split == [1, 2, 1]


class Evil:
    def __reduce__(self):
        return (print, ("pwned",))


def test_load_checkpoint_refuses_pickled_code(tmp_path):
    path = tmp_path / "evil.pt"
    with open(path, "wb") as f:
        torch.save({"config": {}, "ema": {}, "x": Evil()}, f)
    with pytest.raises(pickle.UnpicklingError):
        runtime.load_checkpoint(path, torch.device("cpu"))


def test_load_checkpoint_refuses_bad_structure(tmp_path):
    with pytest.raises(ValueError):
        runtime.load_checkpoint(_save({"config": TINY}, tmp_path / "a.pt"), torch.device("cpu"))
    with pytest.raises(KeyError):
        runtime.load_checkpoint(_save({"config": {**TINY, "evil": 1}, "ema": {}}, tmp_path / "b.pt"), torch.device("cpu"))
    with pytest.raises(ValueError):
        runtime.load_checkpoint(_save({"config": {**TINY, "hidden_size": 100000}, "ema": {}}, tmp_path / "c.pt"), torch.device("cpu"))
    path, model = tiny_ckpt(tmp_path)
    ck = torch.load(path, weights_only=True)
    ck["ema"].pop(next(iter(ck["ema"])))
    with pytest.raises(RuntimeError):  # strict load: a missing key fails
        runtime.load_checkpoint(_save(ck, tmp_path / "d.pt"), torch.device("cpu"))


def test_grid_shape_and_labels():
    imgs = [torch.full((1, 32, 32, 3), v, dtype=torch.uint8) for v in (10, 120, 250)]
    g = runtime.labeled_grid(imgs, ["loops 1", "loops 2", "loops 4"])
    assert g.shape == (1, 32 + 40, 3 * 32 + 2 * 8, 3) and g.dtype == torch.uint8
    assert torch.equal(g[0, 40:, 0:32], imgs[0][0]) and torch.equal(g[0, 40:, 40:72], imgs[1][0])
