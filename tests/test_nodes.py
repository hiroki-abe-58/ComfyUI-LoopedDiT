"""The node pack as ComfyUI loads it: registration and ComfyUI's own validation of the API workflows."""

import asyncio
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"LoopedDiTLoader", "LoopedDiTGenerate", "LoopedDiTLoopSweep", "LoopedDiTUnload"}


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, tmp_path_factory):
    import folder_paths
    import nodes

    models = tmp_path_factory.mktemp("models") / "looped_dit"
    (models / "flan-t5-large").mkdir(parents=True)
    (models / "looped-dit-b16.pt").write_bytes(b"")
    for f in ("config.json", "tokenizer.json", "model.safetensors"):
        (models / "flan-t5-large" / f).write_bytes(b"")
    folder_paths.add_model_folder_path("looped_dit", str(models), is_default=True)
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    folder_paths.filename_list_cache.clear()
    return nodes


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


def test_model_folder_lists_checkpoints_and_text_encoders(comfy_nodes):
    from looped_dit_comfy import paths

    assert "looped-dit-b16.pt" in paths.checkpoints()
    assert "flan-t5-large" in paths.text_encoders()
    with pytest.raises(FileNotFoundError):
        paths.text_encoder_path("..")
    with pytest.raises(FileNotFoundError):
        paths.checkpoint_path("../../secret.pt")


@pytest.mark.parametrize("name", ["looped_dit_generate_api.json", "looped_dit_loop_sweep_api.json"])
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def test_validation_rejects_out_of_range_loops(comfy_nodes):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / "looped_dit_generate_api.json").read_text(encoding="utf-8"))
    prompt["2"]["inputs"]["loops"] = 0
    valid, _, _, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert not valid and "2" in node_errors
