"""Corpus tokenisation, caching, and batching.

Tokenising 60MB of text in pure Python takes minutes, and every ablation run
needs the same tokens. So tokenised corpora are cached to disk as uint16 arrays,
keyed by the tokenizer and character budget that produced them - the cost is
paid once per (tokenizer, corpus) pair rather than once per experiment.

uint16 is sufficient because every vocabulary in this project is under 65,536,
and it halves memory against int32.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig
from .tokenizer import BPETokenizer


def _cache_path(text_path: Path, tokenizer_path: Path, max_chars: int) -> Path:
    key = f"{text_path.name}:{tokenizer_path.name}:{max_chars}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    return text_path.parent / f".cache_{text_path.stem}_{digest}.npy"


def tokenise_corpus(
    text_path: str | Path,
    tokenizer: BPETokenizer,
    tokenizer_path: str | Path,
    max_chars: int,
    verbose: bool = True,
) -> np.ndarray:
    text_path = Path(text_path)
    cache = _cache_path(text_path, Path(tokenizer_path), max_chars)

    if cache.exists():
        tokens = np.load(cache)
        if verbose:
            print(f"  loaded {len(tokens):,} tokens from cache")
        return tokens

    raw = text_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    if verbose:
        print(f"  tokenising {len(raw)/1e6:.1f}M chars (first run; will be cached)...")

    ids = tokenizer.encode(raw)
    tokens = np.asarray(ids, dtype=np.uint16)
    np.save(cache, tokens)
    if verbose:
        print(f"  {len(tokens):,} tokens ({len(raw)/len(tokens):.2f} chars/token)")
    return tokens


class TokenBatcher:
    """Samples fixed-length windows uniformly at random from a token stream.

    Random offsets rather than a fixed epoch order: ablations are compared at
    equal *token budgets* rather than equal epochs, so an index-free sampler
    keeps runs comparable when context length varies.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        context_length: int,
        batch_size: int,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        if len(tokens) <= context_length + 1:
            raise ValueError(
                f"corpus too small: {len(tokens)} tokens for context {context_length}"
            )
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        high = len(self.tokens) - self.context_length - 1
        starts = torch.randint(0, high, (self.batch_size,), generator=self.generator)
        x = torch.stack([self.tokens[s : s + self.context_length] for s in starts])
        # Targets are inputs shifted by one: predicting the next token.
        y = torch.stack([self.tokens[s + 1 : s + self.context_length + 1] for s in starts])
        return x.to(self.device), y.to(self.device)

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.context_length


def build_batchers(
    cfg: DataConfig,
    context_length: int,
    batch_size: int,
    device: torch.device,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[TokenBatcher, TokenBatcher, BPETokenizer]:
    tokenizer = BPETokenizer.load(cfg.tokenizer_path)
    if verbose:
        print(f"tokenizer: {cfg.tokenizer_path} (vocab {len(tokenizer):,})")

    train_tokens = tokenise_corpus(
        cfg.train_path, tokenizer, cfg.tokenizer_path, cfg.max_train_chars, verbose
    )
    val_tokens = tokenise_corpus(
        cfg.val_path, tokenizer, cfg.tokenizer_path, cfg.max_val_chars, verbose
    )

    train = TokenBatcher(train_tokens, context_length, batch_size, device, seed)
    # A separate seed so validation windows are fixed and independent of the
    # training stream, making eval loss comparable across runs.
    val = TokenBatcher(val_tokens, context_length, batch_size, device, seed + 9999)
    return train, val, tokenizer
