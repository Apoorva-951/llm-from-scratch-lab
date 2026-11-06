"""The ablation suite.

Each entry changes exactly one thing against a fixed baseline and states what it
is testing. Runs are compared at an equal token budget, so a difference in final
validation loss is attributable to the field under test rather than to how much
data each run happened to see.

Two entries are deliberate controls rather than improvements:

  no-causal-mask   removes the mask so the model can see future tokens. Loss
                   should collapse - the task becomes copying, not prediction.
                   If it does not collapse, the masking implementation is wrong.
  no-position      removes positional information entirely. Loss should worsen
                   clearly; attention is permutation-invariant without it, so
                   the model can only learn a bag-of-tokens distribution.

Controls are what make the rest of the suite trustworthy: they prove the
measurement can detect a change when one genuinely exists.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig  # noqa: E402

# Held fixed across every run unless the ablation is explicitly about it.
TOKEN_BUDGET = 5_000_000
TOKENIZER = "data/tokenizer-2048.json"
VOCAB = 2048

BASE_MODEL = ModelConfig(
    vocab_size=VOCAB,
    context_length=256,
    emb_dim=384,
    n_heads=6,
    n_layers=6,
    drop_rate=0.0,
    activation="gelu",
    norm="layernorm",
    position="learned",
)

BASE_TRAIN = TrainConfig(
    train_tokens=TOKEN_BUDGET,
    batch_size=16,
    learning_rate=3e-4,
    eval_every_tokens=500_000,
    eval_batches=30,
    seed=1234,
)

BASE_DATA = DataConfig(
    tokenizer_path=TOKENIZER,
    max_train_chars=40_000_000,
    max_val_chars=4_000_000,
)


def _experiment(
    name: str,
    hypothesis: str,
    model_overrides: dict | None = None,
    train_overrides: dict | None = None,
    data_overrides: dict | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        hypothesis=hypothesis,
        model=replace(BASE_MODEL, **(model_overrides or {})),
        train=replace(BASE_TRAIN, **(train_overrides or {})),
        data=replace(BASE_DATA, **(data_overrides or {})),
    )


def build_suite() -> list[ExperimentConfig]:
    suite: list[ExperimentConfig] = [
        _experiment(
            "baseline",
            "GPT-style reference: GELU, LayerNorm, learned positions, 6 heads, 6 layers.",
        ),

        # ---- controls: these should visibly break ----------------------------
        _experiment(
            "control-no-causal-mask",
            "Removing the causal mask lets the model attend to future tokens, "
            "so next-token prediction becomes trivial. Validation loss should "
            "collapse far below baseline. If it does not, masking is broken.",
            {"causal_mask": False},
        ),
        _experiment(
            "control-no-position",
            "Without positional information attention is permutation-invariant, "
            "so the model can only learn token frequencies. Loss should worsen "
            "clearly against baseline.",
            {"position": "none"},
        ),

        # ---- attention heads -------------------------------------------------
        _experiment(
            "heads-1",
            "One head: same parameter count, but a single attention pattern per "
            "layer instead of six specialised subspaces.",
            {"n_heads": 1},
        ),
        _experiment(
            "heads-2", "Two heads - intermediate point in the head-count sweep.", {"n_heads": 2}
        ),
        _experiment(
            "heads-12",
            "Twelve heads: more subspaces, each of dimension 32 instead of 64. "
            "Tests whether head count or per-head width matters more.",
            {"n_heads": 12},
        ),

        # ---- depth -----------------------------------------------------------
        _experiment(
            "layers-2",
            "Shallow: 2 layers. Tests how much depth contributes at fixed width.",
            {"n_layers": 2},
        ),
        _experiment("layers-4", "Intermediate depth.", {"n_layers": 4}),
        _experiment(
            "layers-10",
            "Deeper at the same width. More parameters and more compute per "
            "token, so throughput should drop.",
            {"n_layers": 10},
        ),

        # ---- context length --------------------------------------------------
        _experiment(
            "context-128",
            "Half the context. Attention cost is quadratic in sequence length, "
            "so throughput should rise while long-range structure is lost.",
            {"context_length": 128},
        ),
        _experiment(
            "context-512",
            "Double the context. Quadratic attention cost should reduce "
            "throughput; TinyStories are short, so the benefit may be small.",
            {"context_length": 512},
            {"batch_size": 8},  # halved to keep tokens-per-step comparable
        ),

        # ---- activation ------------------------------------------------------
        _experiment(
            "act-relu",
            "ReLU instead of GELU: a hard zero below zero rather than a smooth "
            "gate. Cheaper, and the classic comparison.",
            {"activation": "relu"},
        ),
        _experiment(
            "act-swiglu",
            "SwiGLU: a gated MLP as used by Llama, with hidden width scaled by "
            "2/3 to keep parameters comparable.",
            {"activation": "swiglu"},
        ),

        # ---- normalisation ---------------------------------------------------
        _experiment(
            "norm-rmsnorm",
            "RMSNorm drops mean subtraction and the shift parameter. Tests "
            "whether re-centring contributes anything at this scale.",
            {"norm": "rmsnorm"},
        ),

        # ---- positional encoding --------------------------------------------
        _experiment(
            "pos-rope",
            "Rotary embeddings inject relative position inside attention and "
            "remove the learned position table entirely.",
            {"position": "rope"},
        ),

        # ---- parameter efficiency -------------------------------------------
        _experiment(
            "tied-embeddings",
            "Sharing input and output embedding weights halves the embedding "
            "budget. Tests whether the saving costs quality.",
            {"tie_embeddings": True},
        ),
        _experiment(
            "gqa-2kv",
            "Grouped-query attention: 6 query heads sharing 2 key/value heads, "
            "cutting KV projection size and memory.",
            {"n_kv_heads": 2},
        ),

        # ---- the combined architecture --------------------------------------
        _experiment(
            "llama-style",
            "All Llama-style choices together: RMSNorm, RoPE, SwiGLU and "
            "grouped-query attention, against the GPT-style baseline.",
            {"norm": "rmsnorm", "position": "rope", "activation": "swiglu", "n_kv_heads": 2},
        ),
    ]
    return suite


def build_vocab_suite() -> list[ExperimentConfig]:
    """Vocabulary sweep - run separately because each size needs its own tokenizer.

    Vocabulary is not a free parameter at this scale: it sets the embedding
    budget and the sequence length a given text occupies, so it trades model
    capacity against compute in both directions at once.
    """
    return [
        _experiment(
            f"vocab-{size}",
            f"Vocabulary {size}: larger vocabularies compress text into fewer "
            f"tokens but spend more parameters on embeddings.",
            {"vocab_size": size},
            None,
            {"tokenizer_path": f"data/tokenizer-{size}.json"},
        )
        for size in (2048, 8192, 16384)
    ]
