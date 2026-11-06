"""Experiment configuration.

Every architectural choice this project ablates is a field on `ModelConfig`, and
every training choice is a field on `TrainConfig`. Nothing that varies between
experiments is hard-coded in the model or the training loop, which is what makes
an ablation a config file rather than a code edit - and what makes results
reproducible from the JSON written alongside them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Activation = Literal["gelu", "relu", "silu", "swiglu"]
Normalisation = Literal["layernorm", "rmsnorm"]
PositionEncoding = Literal["learned", "rope", "none"]


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 256
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6
    drop_rate: float = 0.0
    qkv_bias: bool = False

    # ---- ablation switches -------------------------------------------------
    activation: Activation = "gelu"
    norm: Normalisation = "layernorm"
    position: PositionEncoding = "learned"
    # Grouped-query attention: n_kv_heads < n_heads shares key/value projections
    # across query heads. None means standard multi-head (n_kv_heads == n_heads).
    n_kv_heads: int | None = None
    # Tying input and output embeddings halves the embedding parameter budget,
    # which dominates model size at this scale.
    tie_embeddings: bool = False
    # Deliberately breakable, for the control experiment: with causal masking
    # off the model can attend to future tokens and the language-modelling task
    # becomes trivial. Validation loss should collapse toward zero, which is the
    # point - it demonstrates the mask is what makes the task non-trivial.
    causal_mask: bool = True

    def __post_init__(self) -> None:
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        kv = self.n_kv_heads or self.n_heads
        if self.n_heads % kv != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({kv})"
            )

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.n_heads


@dataclass
class TrainConfig:
    # Token budget rather than epochs: ablations must see exactly the same
    # number of tokens to be comparable, regardless of context length or
    # batch size.
    train_tokens: int = 3_000_000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    eval_every_tokens: int = 250_000
    eval_batches: int = 40
    seed: int = 1234
    device: str = "auto"          # auto | mps | cuda | cpu
    compile_model: bool = False   # torch.compile; unreliable on MPS
    # Where to persist weights. Ablations do not need checkpoints - only the
    # measurements matter and 21 checkpoints would waste gigabytes - but
    # fine-tuning needs a pretrained model to start from, so it is opt-in.
    checkpoint_path: str | None = None


@dataclass
class DataConfig:
    train_path: str = "data/tinystories-train.txt"
    val_path: str = "data/tinystories-valid.txt"
    tokenizer_path: str = "data/tokenizer-8k.json"
    # Cap on characters read, so a run does not depend on corpus size on disk.
    max_train_chars: int = 40_000_000
    max_val_chars: int = 4_000_000


@dataclass
class ExperimentConfig:
    name: str
    # One sentence on what this run is testing, carried into the results file so
    # a plot months later still explains itself.
    hypothesis: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=payload["name"],
            hypothesis=payload.get("hypothesis", ""),
            model=ModelConfig(**payload.get("model", {})),
            train=TrainConfig(**payload.get("train", {})),
            data=DataConfig(**payload.get("data", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
