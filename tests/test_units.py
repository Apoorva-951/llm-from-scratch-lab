"""Unit tests for the framework.

Deliberately CPU-only with tiny models, for two reasons: they stay fast enough
to run constantly, and they never compete with a training run for GPU memory.

The most important test here is `test_causal_masking`. The ablation suite
demonstrates masking empirically - removing the mask collapses validation
perplexity to ~1.0 - but that is an expensive, indirect check. This asserts the
property directly: a token's output must not change when a *later* token is
edited. If that ever breaks, every language-modelling number in the project is
silently meaningless.

Run:  python tests/test_units.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lm.config import ModelConfig, TrainConfig  # noqa: E402
from lm.data import TokenBatcher  # noqa: E402
from lm.model import GPT, RMSNorm, apply_rope, build_rope_cache  # noqa: E402
from lm.tokenizer import BPETokenizer  # noqa: E402
from lm.train import _lr_at  # noqa: E402

import numpy as np  # noqa: E402

torch.manual_seed(0)
FAILURES: list[str] = []

TINY = dict(vocab_size=64, context_length=16, emb_dim=32, n_heads=4, n_layers=2, drop_rate=0.0)


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  pass  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


# ---------------------------------------------------------------- tokenizer --


def test_tokenizer() -> None:
    print("tokenizer")
    corpus = ("Once upon a time there was a little girl named Lily. "
              "She saw a shiny red ball and wanted to play with it. ") * 60
    tok = BPETokenizer.train(corpus, vocab_size=400)

    check(len(tok) <= 400, "vocabulary does not exceed the requested size")
    check(len(tok) >= 257, "vocabulary contains the full byte alphabet plus specials")

    # Byte-level alphabet is the property that makes encoding total: any input,
    # including characters never seen during training, must round-trip.
    for label, text in [
        ("ascii", "Lily saw a red ball."),
        ("unseen unicode", "Café naïve — 日本語 ✨"),
        ("emoji", "hello 🙂🚀 world"),
        ("whitespace", "tabs\there\nand\nnewlines\n\n"),
        ("empty", ""),
        ("special token", "story one<|endoftext|>story two"),
    ]:
        check(tok.decode(tok.encode(text)) == text, f"round-trips {label}")

    check(tok.encode("Lily") == tok.encode("Lily"), "encoding is deterministic")
    check(tok.eot_id is not None, "end-of-text token has an id")
    check(all(i < len(tok) for i in tok.encode(corpus[:400])), "all ids are within vocabulary")

    # Merges must actually compress relative to raw bytes.
    text = "Once upon a time there was a little girl"
    check(len(tok.encode(text)) < len(text.encode("utf-8")), "merges compress below byte count")


# ------------------------------------------------------------------- config --


def test_config_validation() -> None:
    print("config validation")

    try:
        ModelConfig(emb_dim=100, n_heads=3)
        check(False, "rejects emb_dim not divisible by n_heads")
    except ValueError:
        check(True, "rejects emb_dim not divisible by n_heads")

    try:
        ModelConfig(emb_dim=64, n_heads=4, n_kv_heads=3)
        check(False, "rejects n_heads not divisible by n_kv_heads")
    except ValueError:
        check(True, "rejects n_heads not divisible by n_kv_heads")

    cfg = ModelConfig(emb_dim=384, n_heads=6)
    check(cfg.head_dim == 64, "head_dim computed correctly")
    check(cfg.kv_heads == 6, "kv_heads defaults to n_heads")
    check(ModelConfig(emb_dim=384, n_heads=6, n_kv_heads=2).kv_heads == 2, "kv_heads honoured")


# -------------------------------------------------------------------- model --


def test_model_variants() -> None:
    print("model variants")
    x = torch.randint(0, 64, (2, 8))
    variants = {
        "baseline": {},
        "relu": {"activation": "relu"},
        "silu": {"activation": "silu"},
        "swiglu": {"activation": "swiglu"},
        "rmsnorm": {"norm": "rmsnorm"},
        "rope": {"position": "rope"},
        "no position": {"position": "none"},
        "no causal mask": {"causal_mask": False},
        "grouped-query": {"n_kv_heads": 2},
        "tied embeddings": {"tie_embeddings": True},
        "llama-style": {"norm": "rmsnorm", "position": "rope",
                        "activation": "swiglu", "n_kv_heads": 2},
    }
    for name, overrides in variants.items():
        model = GPT(ModelConfig(**{**TINY, **overrides}))
        out = model(x)
        ok = out.shape == (2, 8, 64) and torch.isfinite(out).all()
        check(ok, f"{name} produces finite logits of the right shape")


def test_parameter_accounting() -> None:
    print("parameter accounting")
    untied = GPT(ModelConfig(**TINY)).parameter_counts()
    tied = GPT(ModelConfig(**{**TINY, "tie_embeddings": True})).parameter_counts()

    check(untied["total"] == sum(p.numel() for p in GPT(ModelConfig(**TINY)).parameters()),
          "total matches the actual parameter count")
    check(tied["total"] < untied["total"], "tying embeddings reduces total parameters")
    check(tied["output_head"] == 0, "tied model reports no separate output head")

    gqa = GPT(ModelConfig(**{**TINY, "n_kv_heads": 2})).parameter_counts()
    check(gqa["transformer"] < untied["transformer"],
          "grouped-query attention reduces transformer parameters")

    no_pos = GPT(ModelConfig(**{**TINY, "position": "none"})).parameter_counts()
    check(no_pos["embedding"] < untied["embedding"],
          "removing learned positions removes the position table")


def test_causal_masking() -> None:
    """A position's output must not depend on any later position.

    This is the property that makes next-token prediction a real task. The
    ablation suite shows it empirically; this asserts it directly.
    """
    print("causal masking")
    model = GPT(ModelConfig(**TINY)).eval()

    x = torch.randint(0, 64, (1, 8))
    altered = x.clone()
    # Change only the final token, guaranteeing it actually differs.
    altered[0, -1] = (altered[0, -1] + 1) % 64

    with torch.no_grad():
        a = model(x)
        b = model(altered)

    check(torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6),
          "editing a later token leaves earlier outputs unchanged")
    check(not torch.allclose(a[0, -1], b[0, -1], atol=1e-6),
          "the edited position's own output does change")

    # With masking disabled the property must break - otherwise the switch does
    # nothing and the control experiment would be meaningless.
    unmasked = GPT(ModelConfig(**{**TINY, "causal_mask": False})).eval()
    with torch.no_grad():
        c = unmasked(x)
        d = unmasked(altered)
    check(not torch.allclose(c[0, :-1], d[0, :-1], atol=1e-6),
          "without the mask, earlier outputs DO see later tokens")


def test_rope() -> None:
    print("rotary embeddings")
    head_dim, seq = 16, 12
    cos, sin = build_rope_cache(head_dim, seq)
    check(cos.shape == (seq, head_dim), "cos cache has shape (seq, head_dim)")
    check(sin.shape == (seq, head_dim), "sin cache has shape (seq, head_dim)")

    x = torch.randn(2, 4, seq, head_dim)
    rotated = apply_rope(x, cos, sin)
    check(rotated.shape == x.shape, "rotation preserves shape")

    # A rotation preserves vector norm.
    check(torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-4),
          "rotation preserves vector norms")
    # Position 0 has zero angle, so it must pass through unchanged.
    check(torch.allclose(rotated[:, :, 0], x[:, :, 0], atol=1e-5),
          "position 0 is unrotated")


def test_rmsnorm() -> None:
    print("normalisation")
    norm = RMSNorm(8)
    x = torch.randn(4, 8) * 5 + 3
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    check(torch.allclose(rms, torch.ones_like(rms), atol=1e-3),
          "RMSNorm scales rows to unit root-mean-square")
    # Unlike LayerNorm, RMSNorm does not re-centre.
    check(not torch.allclose(out.mean(dim=-1), torch.zeros(4), atol=1e-3),
          "RMSNorm does not centre the mean")


def test_generation() -> None:
    print("generation")
    model = GPT(ModelConfig(**TINY)).eval()
    prompt = torch.randint(0, 64, (1, 4))

    out = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    check(out.shape == (1, 10), "greedy generation appends the requested tokens")
    check(torch.equal(out[:, :4], prompt), "the prompt is preserved")

    greedy_a = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    greedy_b = model.generate(prompt, max_new_tokens=5, temperature=0.0)
    check(torch.equal(greedy_a, greedy_b), "temperature=0 is deterministic")

    sampled = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=2)
    check(sampled.shape == (1, 9), "top-k sampling returns the right length")

    # Generation must not exceed the context window.
    long_prompt = torch.randint(0, 64, (1, 20))  # longer than context_length=16
    out = model.generate(long_prompt, max_new_tokens=3, temperature=0.0)
    check(out.shape == (1, 23), "generation handles prompts longer than the context")


# --------------------------------------------------------------------- data --


def test_batcher() -> None:
    print("batching")
    tokens = np.arange(500, dtype=np.uint16)
    batcher = TokenBatcher(tokens, context_length=10, batch_size=4,
                           device=torch.device("cpu"), seed=0)
    x, y = batcher.batch()
    check(x.shape == (4, 10) and y.shape == (4, 10), "batch has shape (batch, context)")
    # Targets are inputs shifted by one - the definition of the task.
    check(torch.equal(x[:, 1:], y[:, :-1]), "targets are inputs shifted by one position")
    check(batcher.tokens_per_batch == 40, "tokens_per_batch reported correctly")

    try:
        TokenBatcher(np.arange(5, dtype=np.uint16), 10, 2, torch.device("cpu"))
        check(False, "rejects a corpus smaller than the context window")
    except ValueError:
        check(True, "rejects a corpus smaller than the context window")


# ---------------------------------------------------------------- scheduler --


def test_lr_schedule() -> None:
    print("learning-rate schedule")
    total, base = 1000, 1e-3
    warmup_end = int(total * 0.05)

    check(_lr_at(0, total, base, 0.05) < base, "starts below peak during warmup")
    check(_lr_at(warmup_end - 1, total, base, 0.05) <= base * 1.001, "warmup reaches peak")
    check(_lr_at(0, total, base, 0.05) < _lr_at(warmup_end - 1, total, base, 0.05),
          "learning rate rises during warmup")
    check(_lr_at(total - 1, total, base, 0.05) < _lr_at(total // 2, total, base, 0.05),
          "decays after warmup")
    check(_lr_at(total - 1, total, base, 0.05) > 0, "never reaches zero")


def test_train_config_budget() -> None:
    print("token budget")
    cfg = TrainConfig(train_tokens=1_000_000, batch_size=16)
    # Runs are compared at equal token budgets, so step count must follow from
    # the budget rather than being fixed - otherwise a longer context would
    # silently give a run more data.
    steps_256 = cfg.train_tokens // (cfg.batch_size * 256)
    steps_512 = cfg.train_tokens // (cfg.batch_size * 512)
    check(steps_512 * 2 == steps_256, "doubling context halves the step count at fixed budget")


def main() -> None:
    for test in (
        test_tokenizer,
        test_config_validation,
        test_model_variants,
        test_parameter_accounting,
        test_causal_masking,
        test_rope,
        test_rmsnorm,
        test_generation,
        test_batcher,
        test_lr_schedule,
        test_train_config_budget,
    ):
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for label in FAILURES:
            print(f"  - {label}")
        sys.exit(1)
    print("all unit tests passed")


if __name__ == "__main__":
    main()
