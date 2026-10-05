# Vendored Looped-DiT code

`model.py`, `diffusion.py` and `config.py` are copied **unchanged** from the official repository
https://github.com/OpenSenseNova/Looped-DiT at commit `92a9c1914258361f78e03426588c7c215944b504`
(MIT License, see `LICENSE` in this folder: Copyright (c) 2026 Looped-DiT authors, Copyright (c) 2026 MiniT2I contributors).

| file | sha256 (LF line endings, identical to the upstream blob) |
| --- | --- |
| model.py | bbb12d504e99a8bf6a05d1161427f21111e5ae5ec3aab5eb8ccbe27d0f13c384 |
| diffusion.py | e7096f9dc26259aa7048b697f3bfe30d3104263654c50bf653407c5281873157 |
| config.py | b48ac4679c7f511c03ae715c660de2d4865eb9f5bbfa558913febce772a532d5 |

`tests/test_vendor_integrity.py` checks these hashes. The node code calls `LoopedMMDiT`, `TrainConfig` and
`euler_sample` as they are; it adds no patch to them.
