# Vendored Looped-DiT code

`model.py`, `diffusion.py` and `config.py` are copied from the official repository
https://github.com/OpenSenseNova/Looped-DiT at commit `92a9c1914258361f78e03426588c7c215944b504`
(MIT License, see `LICENSE` in this folder: Copyright (c) 2026 Looped-DiT authors, Copyright (c) 2026 MiniT2I contributors).
`model.py` and `diffusion.py` are unchanged; `config.py` has one change (below).

| file | sha256 (LF line endings) |
| --- | --- |
| model.py | bbb12d504e99a8bf6a05d1161427f21111e5ae5ec3aab5eb8ccbe27d0f13c384 (identical to the upstream blob) |
| diffusion.py | e7096f9dc26259aa7048b697f3bfe30d3104263654c50bf653407c5281873157 (identical to the upstream blob) |
| config.py | fdca98ef60cdf50da3605c3bb9ef613e73a5b594964391e132930f7269101ccc (upstream b48ac4679c7f... with the change) |

`tests/test_vendor_and_tokenization.py` checks these hashes and, with an upstream checkout (CI), compares the files byte for byte
with the pinned commit. The node code calls `LoopedMMDiT`, `TrainConfig` and `euler_sample` as they are.

## The one change (since 0.1.1)

`config.py`: the bodies of `expand_vars()` and `load_yaml()` are replaced by `raise NotImplementedError(...)`, and the file starts
with a one-line change notice (`scripts/vendor_patch.py` makes the change from the upstream file). The two functions read YAML
training configs: `load_yaml()` opens a file by path and `expand_vars()` fills `${VAR}` from environment variables. The Comfy
Registry flagged the environment read (reviewed as arbitrary file read). The nodes build `TrainConfig` with `from_dict()` from the
checkpoint and never called them, so generation is unchanged. The test compares the syntax trees with the upstream file and allows a
difference only in these two functions.
