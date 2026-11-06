"""Training loop and measurement.

Every ablation runs through this one function, and every run is measured
identically: train and validation loss, perplexity, throughput, peak memory,
parameter split and wall time. A result is only comparable if the measurement
is the same, so the measurement lives here rather than in each experiment.

Runs are compared at equal *token budgets*, not equal epochs or steps. Context
length is itself an ablation, and a step processes `batch_size x context_length`
tokens - so equal steps would silently give long-context runs more data.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import TokenBatcher, build_batchers
from .model import GPT


def resolve_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingRecord:
    """Everything measured about one run."""

    name: str
    hypothesis: str
    config: dict
    parameters: dict[str, int]
    device: str
    history: list[dict] = field(default_factory=list)
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_val_perplexity: float = 0.0
    best_val_loss: float = float("inf")
    tokens_per_second: float = 0.0
    wall_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    samples: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=2))


@torch.no_grad()
def evaluate(model: GPT, batcher: TokenBatcher, batches: int) -> float:
    """Mean cross-entropy over a fixed number of validation batches."""
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = batcher.batch()
        logits = model(x)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
    model.train()
    return total / batches


def _lr_at(step: int, total_steps: int, base_lr: float, warmup_frac: float) -> float:
    """Linear warmup then cosine decay to 10% of the peak.

    Warmup matters most for the deeper ablations: without it the first few
    updates on a randomly initialised deep stack can destabilise training, which
    would show up as an architecture effect rather than an optimisation one.
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, progress))))


def train_experiment(cfg: ExperimentConfig, verbose: bool = True) -> TrainingRecord:
    torch.manual_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    train_batcher, val_batcher, tokenizer = build_batchers(
        cfg.data,
        context_length=cfg.model.context_length,
        batch_size=cfg.train.batch_size,
        device=device,
        seed=cfg.train.seed,
        verbose=verbose,
    )

    # The tokenizer is the source of truth for vocabulary size; a mismatch would
    # silently produce an unused or out-of-range embedding row.
    if len(tokenizer) != cfg.model.vocab_size:
        if verbose:
            print(f"  vocab_size {cfg.model.vocab_size} -> {len(tokenizer)} (from tokenizer)")
        cfg.model.vocab_size = len(tokenizer)

    model = GPT(cfg.model).to(device)
    counts = model.parameter_counts()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )

    tokens_per_step = train_batcher.tokens_per_batch
    total_steps = max(1, cfg.train.train_tokens // tokens_per_step)
    eval_every = max(1, cfg.train.eval_every_tokens // tokens_per_step)

    record = TrainingRecord(
        name=cfg.name,
        hypothesis=cfg.hypothesis,
        config=cfg.to_dict(),
        parameters=counts,
        device=str(device),
    )

    if verbose:
        print(
            f"\n{cfg.name}\n"
            f"  params  total {counts['total']/1e6:.2f}M "
            f"(transformer {counts['transformer']/1e6:.2f}M, "
            f"embedding {counts['embedding']/1e6:.2f}M)\n"
            f"  budget  {cfg.train.train_tokens:,} tokens "
            f"= {total_steps:,} steps of {tokens_per_step:,}\n"
        )

    if device.type == "mps":
        torch.mps.empty_cache()

    model.train()
    started = time.time()
    tokens_seen = 0
    running_loss = 0.0
    loss_steps = 0

    for step in range(total_steps):
        for group in optimiser.param_groups:
            group["lr"] = _lr_at(step, total_steps, cfg.train.learning_rate, cfg.train.warmup_frac)

        x, y = train_batcher.batch()
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimiser.step()

        tokens_seen += tokens_per_step
        running_loss += loss.item()
        loss_steps += 1

        if (step + 1) % eval_every == 0 or step == total_steps - 1:
            train_loss = running_loss / max(1, loss_steps)
            running_loss = 0.0
            loss_steps = 0
            val_loss = evaluate(model, val_batcher, cfg.train.eval_batches)
            elapsed = time.time() - started

            record.history.append(
                {
                    "step": step + 1,
                    "tokens_seen": tokens_seen,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    # Perplexity is exp(cross-entropy): the effective number of
                    # equally likely next tokens the model is choosing between.
                    "val_perplexity": round(math.exp(min(20, val_loss)), 3),
                    "elapsed_s": round(elapsed, 1),
                }
            )
            record.best_val_loss = min(record.best_val_loss, val_loss)

            if verbose:
                print(
                    f"  {tokens_seen/1e6:5.2f}M tok  "
                    f"train {train_loss:6.3f}  val {val_loss:6.3f}  "
                    f"ppl {math.exp(min(20, val_loss)):8.2f}  "
                    f"{elapsed:5.0f}s",
                    flush=True,
                )

    if device.type == "mps":
        torch.mps.synchronize()
    record.wall_seconds = round(time.time() - started, 1)
    record.tokens_per_second = round(tokens_seen / record.wall_seconds, 1)
    record.final_train_loss = record.history[-1]["train_loss"] if record.history else 0.0
    record.final_val_loss = record.history[-1]["val_loss"] if record.history else 0.0
    record.final_val_perplexity = record.history[-1]["val_perplexity"] if record.history else 0.0

    if device.type == "mps":
        record.peak_memory_mb = round(torch.mps.current_allocated_memory() / 1e6, 1)
    elif device.type == "cuda":
        record.peak_memory_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    record.samples = generate_samples(model, tokenizer, device)

    if cfg.train.checkpoint_path:
        path = Path(cfg.train.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The model config travels with the weights: a checkpoint that cannot
        # say what architecture produced it is not loadable later.
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": cfg.model.__dict__,
                "tokenizer_path": cfg.data.tokenizer_path,
                "final_val_loss": record.final_val_loss,
            },
            path,
        )
        if verbose:
            print(f"  saved checkpoint to {path}")

    if verbose:
        print(
            f"  done in {record.wall_seconds:.0f}s  "
            f"{record.tokens_per_second:,.0f} tok/s  "
            f"final val ppl {record.final_val_perplexity:.2f}"
        )
    return record


@torch.no_grad()
def generate_samples(
    model: GPT,
    tokenizer,
    device: torch.device,
    prompts: list[str] | None = None,
    max_new_tokens: int = 60,
) -> list[str]:
    """Qualitative samples stored with every run.

    Loss numbers say a model improved; only samples say whether the text is
    coherent, and the two can disagree in informative ways.
    """
    prompts = prompts or ["Once upon a time", "Lily saw a", "The little boy was very"]
    outputs: list[str] = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = model.generate(
            ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40,
            eos_id=tokenizer.eot_id,
        )
        outputs.append(tokenizer.decode(generated[0].tolist()))
    return outputs
