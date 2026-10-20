"""The node pack's one documented change to the vendored upstream code (maintainer tool and test helper).

    python scripts/vendor_patch.py --upstream <OpenSenseNova/Looped-DiT checkout at the pinned commit>

`looped_dit/config.py`: `expand_vars()` substitutes `${VAR}` in YAML training configs with environment variables, and `load_yaml()`
opens a config file by path. Both serve upstream training only; the nodes build `TrainConfig` with `from_dict()` from the checkpoint
and never call them. The Comfy Registry flagged the environment read (reviewed as arbitrary file read). Their bodies are replaced by
an error, and the file starts with a change notice (MIT; kept with the upstream LICENSE). `model.py` and `diffusion.py` are
byte-identical to upstream. tests/test_vendor_and_tokenization.py applies `patch()` to the upstream checkout in CI and compares the
bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "looped_dit_comfy" / "vendor"
PATCHED = {"config.py"}
NOTICE = "# This file was changed by ComfyUI-LoopedDiT; looped_dit_comfy/vendor/UPSTREAM.md describes the change.\n"
_OLD = '''def expand_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} in strings; unknown variables raise."""
    if isinstance(value, str):

        def substitute(match: re.Match) -> str:
            name = match.group(1)
            if name in os.environ:
                return os.environ[name]
            if name in DEFAULT_ROOTS:
                return str(DEFAULT_ROOTS[name])
            raise KeyError(f"undefined variable ${{{name}}} in {value!r}")

        return re.sub(r"\\$\\{(\\w+)\\}", substitute, value)
    if isinstance(value, list):
        return [expand_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_vars(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return expand_vars(data)
'''
_NEW = '''def expand_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} in strings (training configs; not part of the ComfyUI node pack)."""
    raise NotImplementedError("expand_vars is used by training configs and is not included in ComfyUI-LoopedDiT")


def load_yaml(path: str | Path) -> dict[str, Any]:
    raise NotImplementedError("load_yaml reads training configs and is not included in ComfyUI-LoopedDiT")
'''


def patch(rel: str, text: str) -> str:
    if rel != "config.py":
        raise KeyError(rel)
    if text.count(_OLD) != 1:
        raise ValueError(f"{rel}: expand_vars() / load_yaml() differ from the pinned upstream version")
    return NOTICE + text.replace(_OLD, _NEW)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    args = ap.parse_args(argv)
    for rel in sorted(PATCHED):
        data = patch(rel, (args.upstream / "looped_dit" / rel).read_bytes().decode("utf-8")).encode("utf-8")
        (VENDOR / rel).write_bytes(data)
        print("wrote", rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
