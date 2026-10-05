import hashlib
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "looped_dit_comfy" / "vendor"
EXPECTED = {
    "model.py": "bbb12d504e99a8bf6a05d1161427f21111e5ae5ec3aab5eb8ccbe27d0f13c384",
    "diffusion.py": "e7096f9dc26259aa7048b697f3bfe30d3104263654c50bf653407c5281873157",
    "config.py": "b48ac4679c7f511c03ae715c660de2d4865eb9f5bbfa558913febce772a532d5",
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_vendored_file_is_unmodified(name):
    assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_vendored_file_equals_upstream_checkout(upstream_dir, name):
    assert (VENDOR / name).read_bytes() == (Path(upstream_dir) / "looped_dit" / name).read_bytes()


def test_tokenization_equals_transformers4(tokenizer_json):
    """317 prompts (edge cases + generated whitespace/punctuation/unicode mixes) recorded with transformers 4.57.6."""
    from looped_dit_comfy.tokenization import T5Tokenization

    fx = json.loads((ROOT / "tests" / "fixtures" / "t5_tokens_transformers4.json").read_text(encoding="utf-8"))
    tok = T5Tokenization(tokenizer_json, fx["prompt_length"])
    ids, mask = tok([it["prompt"] for it in fx["items"]])
    assert ids.shape == mask.shape == (len(fx["items"]), 256) and ids.dtype == mask.dtype == torch.long
    for k, it in enumerate(fx["items"]):
        n = int(mask[k].sum())
        assert ids[k, :n].tolist() == it["ids"], repr(it["prompt"])
        assert mask[k, :n].all() and not mask[k, n:].any() and not ids[k, n:].any()


def test_trailing_whitespace_keeps_the_trained_token(tokenizer_json):
    from looped_dit_comfy.tokenization import T5Tokenization

    ids, mask = T5Tokenization(tokenizer_json, 256)(["a cat "])
    assert ids[0, : int(mask[0].sum())].tolist() == [3, 9, 1712, 3, 1]  # transformers 5.x gives [3, 9, 1712, 1]
