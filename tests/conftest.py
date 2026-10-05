"""Test setup. References come from environment variables, never from this repository:

LOOPED_DIT_COMFYUI_DIR    ComfyUI checkout (v0.38.0 or later)
LOOPED_DIT_UPSTREAM_DIR   OpenSenseNova/Looped-DiT checkout (the commit in looped_dit_comfy/vendor/UPSTREAM.md)
LOOPED_DIT_T5_TOKENIZER   google/flan-t5-large tokenizer.json (revision 0613663, 2.4 MB)
LOOPED_DIT_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
No model weights are used: the model tests run tiny random-initialized models on the CPU.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQUIRE = os.environ.get("LOOPED_DIT_REQUIRE_REFERENCES") == "1"


def _dir(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _dir("LOOPED_DIT_COMFYUI_DIR")
UPSTREAM = _dir("LOOPED_DIT_UPSTREAM_DIR")
TOKENIZER = _dir("LOOPED_DIT_T5_TOKENIZER")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (LOOPED_DIT_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (LOOPED_DIT_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def upstream_dir():
    return _need(UPSTREAM, "Looped-DiT checkout (LOOPED_DIT_UPSTREAM_DIR)")


@pytest.fixture(scope="session")
def tokenizer_json():
    return _need(TOKENIZER, "flan-t5-large tokenizer.json (LOOPED_DIT_T5_TOKENIZER)")
