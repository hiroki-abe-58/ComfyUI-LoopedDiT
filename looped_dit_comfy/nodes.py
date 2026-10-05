"""ComfyUI nodes for Looped-DiT (in-process; no subprocess, no download at run time)."""

from __future__ import annotations

import json

from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

LoopedDiT = io.Custom("LOOPED_DIT")

LOOPS_HELP = ("Loop depth = how many times the shared middle blocks run inside ONE denoising step "
              "(6 + 5 x loops + 6 blocks per model pass for B/16). It is not the number of sampling steps. "
              "The models were trained at 4.")


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(total)

    def before_forward():
        mm.throw_exception_if_processing_interrupted()
        bar.update(1)

    return before_forward


def _get(model: dict) -> runtime.Loaded:
    return runtime.acquire(model["checkpoint"], model["text_encoder"], model["device"], model["verify_hashes"])


class LoopedDiTLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoopedDiTLoader",
            display_name="Looped-DiT Loader",
            category="loaders/looped-dit",
            description=("Loads a Looped-DiT checkpoint (EMA weights, bfloat16) and its FLAN-T5-Large text encoder from "
                         "models/looped_dit. One model stays resident; loading another frees it."),
            inputs=[
                io.Combo.Input("checkpoint", options=paths.checkpoints(), tooltip="A .pt file in models/looped_dit"),
                io.Combo.Input("text_encoder", options=paths.text_encoders(),
                               tooltip="A folder in models/looped_dit with config.json, tokenizer.json, model.safetensors"),
                io.Boolean.Input("verify_hashes", default=True,
                                 tooltip="SHA-256 of the checkpoint and text encoder, compared with the official releases (once per file)"),
            ],
            outputs=[LoopedDiT.Output(display_name="model"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, checkpoint, text_encoder, verify_hashes) -> io.NodeOutput:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Looped-DiT for ComfyUI needs a CUDA GPU (CPU and Apple MPS are not supported in this version)")
        handle = {"checkpoint": paths.checkpoint_path(checkpoint), "text_encoder": paths.text_encoder_path(text_encoder),
                  "device": f"cuda:{torch.cuda.current_device()}", "verify_hashes": bool(verify_hashes)}
        loaded = _get(handle)
        return io.NodeOutput(handle, json.dumps(loaded.info, indent=1))


def _common_inputs():
    return [
        LoopedDiT.Input("model"),
        io.String.Input("prompt", multiline=True, default="a red cube on top of a blue sphere"),
        io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
    ]


def _tail_inputs():
    return [
        io.Int.Input("steps", default=100, min=1, max=runtime.MAX_STEPS, tooltip="Euler steps (official: 100)"),
        io.Float.Input("cfg", default=6.0, min=1.0, max=30.0, step=0.1,
                       tooltip="Classifier-free guidance (official: 6.0). The unconditional pass uses the same text with a zero mask."),
    ]


class LoopedDiTGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoopedDiTGenerate",
            display_name="Looped-DiT Generate",
            category="sampling/looped-dit",
            description="Text to image with the official Euler sampler (pixel space, 512x512 for the released models).",
            inputs=_common_inputs() + [io.Int.Input("loops", default=4, min=1, max=runtime.MAX_LOOPS, tooltip=LOOPS_HELP)] + _tail_inputs(),
            outputs=[io.Image.Output(display_name="image"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, prompt, seed, loops, steps, cfg) -> io.NodeOutput:
        loaded = _get(model)
        passes = 2 if float(cfg) != 1.0 else 1
        out = runtime.generate(loaded, prompt, seed, loops, steps, cfg, before_forward=_progress(int(steps) * passes))
        report = {**out["report"], "model": {k: loaded.info.get(k) for k in ("checkpoint_file", "checkpoint_release", "text_encoder_release")}}
        return io.NodeOutput(out["image"], json.dumps(report, indent=1))


class LoopedDiTLoopSweep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoopedDiTLoopSweep",
            display_name="Looped-DiT Loop Sweep",
            category="sampling/looped-dit",
            description=("Runs the same prompt, seed, initial noise, steps and CFG once per loop depth, each as a full "
                         "sampling run, one after another. Returns every image, a labeled side-by-side grid and a JSON report. "
                         "(The official sample.py has the same comparison with --loops 1 2 3 4.)"),
            inputs=_common_inputs() + [io.String.Input("loops", default="1, 2, 4", tooltip="Comma-separated loop depths, at most 8. " + LOOPS_HELP)]
            + _tail_inputs(),
            outputs=[io.Image.Output(display_name="images"), io.Image.Output(display_name="grid"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, prompt, seed, loops, steps, cfg) -> io.NodeOutput:
        import torch

        depths = runtime.parse_loops(loops)
        loaded = _get(model)
        passes = 2 if float(cfg) != 1.0 else 1
        before = _progress(int(steps) * passes * len(depths))
        results = []
        for n in depths:  # a cancel raises inside generate() and stops the remaining depths
            results.append(runtime.generate(loaded, prompt, seed, n, steps, cfg, before_forward=before))
        reports = [r["report"] for r in results]
        noise = {r["noise_sha"] for r in reports}
        labels = [f"loops {r['loops']}  |  {r['double_stream_blocks_per_forward']} blocks/pass  |  {r['steps']} steps x{r['passes_per_step']}"
                  for r in reports]
        grid = runtime.uint8_to_comfy_image(runtime.labeled_grid([r["uint8"] for r in results], labels))
        summary = {"same_initial_noise": len(noise) == 1, "noise_sha": sorted(noise),
                   "model": {k: loaded.info.get(k) for k in ("checkpoint_file", "checkpoint_release", "text_encoder_release")},
                   "runs": reports}
        images = torch.cat([r["image"] for r in results], dim=0)
        return io.NodeOutput(images, grid, json.dumps(summary, indent=1))


class LoopedDiTUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoopedDiTUnload",
            display_name="Looped-DiT Unload",
            category="loaders/looped-dit",
            description="Frees the resident Looped-DiT model and text encoder (GPU and RAM).",
            inputs=[io.Image.Input("after", optional=True, tooltip="Optional: connect an image so this runs after it")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        import torch

        was = runtime.resident()
        freed = runtime.unload()
        report = {"unloaded": freed, "was": was,
                  "cuda_allocated_mib": torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else None}
        text = json.dumps(report, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class LoopedDiTExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoopedDiTLoader, LoopedDiTGenerate, LoopedDiTLoopSweep, LoopedDiTUnload]


async def comfy_entrypoint() -> LoopedDiTExtension:
    paths.register()
    return LoopedDiTExtension()
