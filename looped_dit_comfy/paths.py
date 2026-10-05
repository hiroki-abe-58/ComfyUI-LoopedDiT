"""Model locations: `<ComfyUI models>/looped_dit` (plus any path an administrator adds for `looped_dit` in
extra_model_paths.yaml). Checkpoints are the `.pt` files there; text encoders are sub-folders holding config.json,
tokenizer.json and model.safetensors (e.g. `looped_dit/flan-t5-large/`). Workflows choose names from these lists only."""

from __future__ import annotations

import os

FOLDER = "looped_dit"


def register() -> None:
    import folder_paths

    default = os.path.join(folder_paths.models_dir, FOLDER)
    folder_paths.add_model_folder_path(FOLDER, default, is_default=False)
    paths, _ = folder_paths.folder_names_and_paths[FOLDER]
    folder_paths.folder_names_and_paths[FOLDER] = (paths, {".pt"})


def checkpoints() -> list[str]:
    import folder_paths

    return folder_paths.get_filename_list(FOLDER)


def checkpoint_path(name: str) -> str:
    import folder_paths

    if name not in checkpoints():
        raise FileNotFoundError(f"checkpoint {name!r} is not in the looped_dit model folder")
    path = folder_paths.get_full_path(FOLDER, name)
    if path is None:
        raise FileNotFoundError(name)
    return path


def text_encoders() -> list[str]:
    import folder_paths

    from .runtime import T5_FILES

    found = set()
    for base in folder_paths.get_folder_paths(FOLDER):
        if not os.path.isdir(base):
            continue
        for entry in os.scandir(base):
            if entry.is_dir() and all(os.path.isfile(os.path.join(entry.path, f)) for f in T5_FILES):
                found.add(entry.name)
    return sorted(found)


def text_encoder_path(name: str) -> str:
    import folder_paths

    if name not in text_encoders():
        raise FileNotFoundError(f"text encoder folder {name!r} is not in the looped_dit model folder")
    for base in folder_paths.get_folder_paths(FOLDER):
        cand = os.path.realpath(os.path.join(base, name))
        if os.path.dirname(cand) == os.path.realpath(base) and os.path.isdir(cand):
            return cand
    raise FileNotFoundError(name)
