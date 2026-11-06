"""LoRA: low-rank adaptation.

The idea in one line: instead of updating a weight matrix W directly, freeze it
and learn a low-rank correction, so the forward pass becomes

    h = W x + (alpha / r) * B(A x)

where A is (r x in) and B is (out x r) with r much smaller than either
dimension. Only A and B are trained. B starts at zero, so the adapted model is
*identical* to the base model at step zero - fine-tuning begins from exactly the
pretrained behaviour rather than a perturbed version of it.

Why it matters: a 384x384 projection has 147,456 parameters; its rank-8 adapter
has 384*8 + 8*384 = 6,144, about 4%. Optimiser state scales with *trainable*
parameters, so the memory saving is larger than the parameter saving - Adam
keeps two moments per trainable weight.

This module exists to be measured, not just used: the experiment compares LoRA
against full fine-tuning on trainable parameters, memory, speed, quality on the
new domain, and how much of the original domain is forgotten.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank correction."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.rank = rank
        # Scaling decouples the learning rate from the choice of rank, so
        # changing r does not silently change the effective step size.
        self.scaling = alpha / rank

        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A is initialised like a normal layer, B at zero: their product is zero,
        # so the wrapped module starts out mathematically identical to the base.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + update * self.scaling

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
) -> nn.Module:
    """Freeze the model and wrap the targeted projections with adapters.

    Query and value projections are the default targets, following the LoRA
    paper's finding that adapting those two gives most of the benefit at a
    fraction of the parameters. Key and output projections, and the MLP, are
    left frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1

    if replaced == 0:
        raise ValueError(f"no modules matched targets {targets}; nothing to adapt")
    return model


def parameter_summary(model: nn.Module) -> dict[str, float]:
    """Trainable vs frozen parameter accounting.

    `optimiser_state_mb` is the number that explains why LoRA saves memory in
    practice: AdamW stores two fp32 moments per *trainable* parameter, so
    optimiser state is roughly 8 bytes per trainable weight - often larger than
    the adapters themselves.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(100 * trainable / total, 3) if total else 0.0,
        "optimiser_state_mb": round(trainable * 8 / 1e6, 2),
    }


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold each adapter back into its base weight and unwrap it.

    After merging, inference costs exactly what the base model costs - the
    adapter adds no latency. This is the property that makes LoRA practical for
    serving, as distinct from merely cheap to train.
    """
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    delta = (child.lora_b @ child.lora_a) * child.scaling
                    child.base.weight += delta
                setattr(module, name, child.base)
    return model
