"""Looped-DiT for ComfyUI (unofficial integration of https://github.com/OpenSenseNova/Looped-DiT)."""

import logging

try:
    from .looped_dit_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[Looped-DiT] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
