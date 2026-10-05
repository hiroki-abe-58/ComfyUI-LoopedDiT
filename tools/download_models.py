"""One-time setup: download the official Looped-DiT checkpoint and FLAN-T5-Large into <ComfyUI>/models/looped_dit,
at pinned revisions, and verify every file's SHA-256. The ComfyUI nodes never download anything.

    python tools/download_models.py --models-dir <ComfyUI>/models            # B/16 (about 1.0 GB) + FLAN-T5-Large (about 3.2 GB)
    python tools/download_models.py --models-dir <ComfyUI>/models --b32      # also B/32 (about 1.0 GB)

Run it with the Python of your ComfyUI environment (huggingface_hub is installed with transformers).
Licenses: Looped-DiT code and weights MIT; FLAN-T5-Large Apache-2.0 (see NOTICE).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

FILES = {
    "b16": ("sensenova/Looped-DiT-B16", "7436a1994bc9d0d8d176525d0dad2be458fbe4df",
            {"looped-dit-b16.pt": "2da674002833ed6cb7a7e9cceb3a680700ca251d06b6d1b117e083b8c6811eba"}, ""),
    "b32": ("sensenova/Looped-DiT-B32", "83a623e2d2e34d47e4f118dae84726901c5640df",
            {"looped-dit-b32.pt": "175f89f29c024732763dc4e36fd76aa581d94c96dfea06a4bf41d9c5552e4ecd"}, ""),
    "t5": ("google/flan-t5-large", "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
           {"model.safetensors": "a9dd06ce490f139af36e9eb77dd3758b4fd07a08a73d5a1abe5ff2591e2d388e",
            "tokenizer.json": "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a",
            "config.json": "bfa5beeb5a4630a97f043f071b9b5d858c842604cff5db874680f33b56090c8c"}, "flan-t5-large"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str, target_root: Path) -> None:
    from huggingface_hub import hf_hub_download

    repo, rev, files, sub = FILES[key]
    target = target_root / sub if sub else target_root
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        dest = target / name
        if dest.exists() and sha256(dest) == expected:
            print(f"ok (already present): {dest}")
            continue
        with tempfile.TemporaryDirectory(dir=target) as tmp:
            got = Path(hf_hub_download(repo, name, revision=rev, local_dir=tmp))
            actual = sha256(got)
            if actual != expected:
                raise SystemExit(f"SHA-256 mismatch for {repo}/{name}: {actual} (expected {expected}); nothing installed")
            shutil.move(str(got), dest)
        print(f"ok: {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", required=True, help="the ComfyUI models folder (its looped_dit subfolder is used)")
    ap.add_argument("--b32", action="store_true", help="also download Looped-DiT B/32")
    a = ap.parse_args()
    root = Path(os.path.expanduser(a.models_dir)) / "looped_dit"
    for key in ["b16", "t5"] + (["b32"] if a.b32 else []):
        fetch(key, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
