"""FLAN-T5 tokenization as the Looped-DiT models were trained with.

Upstream pins transformers < 5 because transformers 5.x tokenizes trailing whitespace differently (for example
"a cat " -> [3, 9, 1712, 1] instead of [3, 9, 1712, 3, 1]). ComfyUI installs transformers 5.x, so this module drives
the `tokenizers` library directly with the checkpoint's own tokenizer.json, which gives exactly the token IDs and
attention mask of transformers 4.x T5TokenizerFast (pad/truncate to prompt_length, right side, </s> appended).
tests/test_tokenization.py holds expected IDs recorded with transformers 4.57.6.
"""

from __future__ import annotations

from pathlib import Path

import torch


class T5Tokenization:
    def __init__(self, tokenizer_json: str | Path, prompt_length: int):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer_json))
        tok.enable_truncation(max_length=prompt_length)
        tok.enable_padding(direction="right", pad_id=0, pad_type_id=0, pad_token="<pad>", length=prompt_length)
        self.tokenizer = tok
        self.prompt_length = prompt_length

    def __call__(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer.encode_batch(list(prompts))
        ids = torch.tensor([e.ids for e in enc], dtype=torch.long)
        mask = torch.tensor([e.attention_mask for e in enc], dtype=torch.long)
        return ids, mask
