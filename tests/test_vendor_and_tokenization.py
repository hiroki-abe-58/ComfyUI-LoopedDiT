import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "looped_dit_comfy" / "vendor"
EXPECTED = {
    "model.py": "bbb12d504e99a8bf6a05d1161427f21111e5ae5ec3aab5eb8ccbe27d0f13c384",
    "diffusion.py": "e7096f9dc26259aa7048b697f3bfe30d3104263654c50bf653407c5281873157",
    "config.py": "fdca98ef60cdf50da3605c3bb9ef613e73a5b594964391e132930f7269101ccc",  # scripts/vendor_patch.py applied
}
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_vendored_file_matches_its_hash(name):
    assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_vendored_file_equals_upstream_checkout(upstream_dir, name):
    """Byte-identical to the pinned upstream file; config.py equals it with scripts/vendor_patch.py applied."""
    import vendor_patch

    original = (Path(upstream_dir) / "looped_dit" / name).read_bytes()
    expected = vendor_patch.patch(name, original.decode("utf-8")).encode("utf-8") if name in vendor_patch.PATCHED else original
    assert (VENDOR / name).read_bytes() == expected


def test_the_patch_changes_only_the_training_config_loaders(upstream_dir):
    before = ast.parse((Path(upstream_dir) / "looped_dit" / "config.py").read_text(encoding="utf-8"))
    after = ast.parse((VENDOR / "config.py").read_text(encoding="utf-8"))
    changed = [getattr(a, "name", None) for a, b in zip(before.body, after.body) if ast.dump(a) != ast.dump(b)]
    assert len(before.body) == len(after.body) and changed == ["expand_vars", "load_yaml"]


PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "looped_dit_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess\.|import subprocess|os\.system"
                       r"|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__|weights_only\s*=\s*False|pickle\.|import pickle")


def test_package_has_no_environment_network_process_or_unpickling_code():
    """The categories of the Comfy Registry's scan and review, in code (not comments). scripts/ and tests/ are not part of the
    Registry package."""
    hits = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}" for p in PACKAGE_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line.split("#", 1)[0])]
    assert not hits, "\n".join(hits)


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
