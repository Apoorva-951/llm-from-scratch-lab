"""LoRA versus full fine-tuning, measured.

A model pretrained on TinyStories is adapted to nineteenth-century literary
prose (Sherlock Holmes, Pride and Prejudice, Frankenstein) - same language,
very different register and vocabulary. Two adaptation strategies are compared
under an identical token budget:

    full  - every weight trainable
    lora  - base frozen, rank-8 adapters on the query and value projections

Four things are measured, and the fourth is the one that matters:

    trainable parameters and optimiser memory   the advertised saving
    throughput                                  whether it is actually faster
    loss on the NEW domain                      does the saving cost quality
    loss on the ORIGINAL domain, after          catastrophic forgetting

The last is the interesting comparison. Full fine-tuning rewrites every weight
and typically destroys the original capability; LoRA leaves the base weights
untouched, so the pretrained behaviour survives. Measuring that difference is
cheap here because the evaluation harness already exists.

    python experiments/finetune.py --pretrain     # train and save the base model
    python experiments/finetune.py                # run the comparison
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig  # noqa: E402
from lm.data import TokenBatcher, tokenise_corpus  # noqa: E402
from lm.lora import apply_lora, parameter_summary  # noqa: E402
from lm.model import GPT  # noqa: E402
from lm.tokenizer import BPETokenizer  # noqa: E402
from lm.train import evaluate, generate_samples, resolve_device, train_experiment  # noqa: E402

CHECKPOINT = ROOT / "results" / "checkpoints" / "base.pt"
RESULTS = ROOT / "results" / "finetune.json"

TOKENIZER = "data/tokenizer-2048.json"
FINETUNE_TOKENS = 1_500_000
BATCH_SIZE = 16
CONTEXT = 256


def pretrain() -> None:
    """Train the base model and keep the weights.

    The ablation suite deliberately discards checkpoints - only measurements
    matter there, and 21 checkpoints would waste gigabytes. Fine-tuning needs
    one, so it is trained explicitly.
    """
    cfg = ExperimentConfig(
        name="base-for-finetuning",
        hypothesis="Pretrain on TinyStories and keep the weights for adaptation.",
        model=ModelConfig(vocab_size=2048, context_length=CONTEXT, emb_dim=384,
                          n_heads=6, n_layers=6),
        train=TrainConfig(train_tokens=5_000_000, batch_size=BATCH_SIZE,
                          eval_every_tokens=1_000_000, eval_batches=30,
                          checkpoint_path=str(CHECKPOINT)),
        data=DataConfig(tokenizer_path=TOKENIZER),
    )
    record = train_experiment(cfg)
    print(f"\nbase model ready: val loss {record.final_val_loss:.3f}")


def _load_base(device: torch.device) -> tuple[GPT, ModelConfig]:
    if not CHECKPOINT.exists():
        raise SystemExit(
            f"no checkpoint at {CHECKPOINT}\nrun: python experiments/finetune.py --pretrain"
        )
    blob = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    return model, cfg


def _finetune(
    strategy: str,
    device: torch.device,
    new_train: TokenBatcher,
    new_val: TokenBatcher,
    original_val: TokenBatcher,
    tokenizer: BPETokenizer,
) -> dict:
    model, _ = _load_base(device)

    if strategy == "lora":
        model = apply_lora(model, rank=8, alpha=16.0)
    model.to(device)

    params = parameter_summary(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=3e-4 if strategy == "full" else 1e-3,
                                  weight_decay=0.01)

    # Both strategies see exactly the same tokens, so any difference in outcome
    # is attributable to the method rather than to the amount of adaptation.
    tokens_per_step = new_train.tokens_per_batch
    steps = max(1, FINETUNE_TOKENS // tokens_per_step)

    history = []
    model.train()
    started = time.time()
    tokens_seen = 0

    for step in range(steps):
        x, y = new_train.batch()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimiser.step()
        tokens_seen += tokens_per_step

        if (step + 1) % max(1, steps // 5) == 0 or step == steps - 1:
            new_loss = evaluate(model, new_val, 25)
            old_loss = evaluate(model, original_val, 25)
            history.append({
                "tokens_seen": tokens_seen,
                "new_domain_loss": round(new_loss, 4),
                "original_domain_loss": round(old_loss, 4),
            })
            print(f"    {tokens_seen/1e6:4.2f}M  new {new_loss:6.3f}  original {old_loss:6.3f}",
                  flush=True)

    if device.type == "mps":
        torch.mps.synchronize()
    wall = time.time() - started

    return {
        "strategy": strategy,
        "parameters": params,
        "history": history,
        "final_new_domain_loss": history[-1]["new_domain_loss"],
        "final_original_domain_loss": history[-1]["original_domain_loss"],
        "wall_seconds": round(wall, 1),
        "tokens_per_second": round(tokens_seen / wall, 1),
        "samples": generate_samples(model, tokenizer, device,
                                    prompts=["It was a", "The gentleman"]),
    }


def compare() -> None:
    device = resolve_device()
    tokenizer = BPETokenizer.load(ROOT / TOKENIZER)

    print(f"device: {device}\nloading corpora...")
    new_tokens = tokenise_corpus(ROOT / "data/gutenberg.txt", tokenizer, TOKENIZER, 1_700_000)
    original_tokens = tokenise_corpus(
        ROOT / "data/tinystories-valid.txt", tokenizer, TOKENIZER, 4_000_000
    )

    split = int(len(new_tokens) * 0.9)
    new_train = TokenBatcher(new_tokens[:split], CONTEXT, BATCH_SIZE, device, seed=7)
    new_val = TokenBatcher(new_tokens[split:], CONTEXT, BATCH_SIZE, device, seed=8)
    original_val = TokenBatcher(original_tokens, CONTEXT, BATCH_SIZE, device, seed=9)

    # Where the base model stands before any adaptation - the reference both
    # strategies move away from.
    base, _ = _load_base(device)
    before = {
        "new_domain_loss": round(evaluate(base, new_val, 25), 4),
        "original_domain_loss": round(evaluate(base, original_val, 25), 4),
    }
    print(f"\nbefore adaptation: new domain {before['new_domain_loss']:.3f}, "
          f"original domain {before['original_domain_loss']:.3f}\n")
    del base

    results = {"before_adaptation": before, "runs": {}}
    for strategy in ("full", "lora"):
        print(f"  {strategy} fine-tuning")
        results["runs"][strategy] = _finetune(
            strategy, device, new_train, new_val, original_val, tokenizer
        )
        print()

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    _report(results)
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")


def _report(results: dict) -> None:
    before = results["before_adaptation"]
    full = results["runs"]["full"]
    lora = results["runs"]["lora"]

    print("=" * 88)
    print("LoRA vs FULL FINE-TUNING")
    print("=" * 88)
    print(f"\n{'metric':34s} {'full':>16s} {'lora':>16s}")
    print("-" * 88)

    rows = [
        ("trainable parameters", f"{full['parameters']['trainable']:,}",
         f"{lora['parameters']['trainable']:,}"),
        ("trainable %", f"{full['parameters']['trainable_pct']:.3f}%",
         f"{lora['parameters']['trainable_pct']:.3f}%"),
        ("optimiser state (MB)", f"{full['parameters']['optimiser_state_mb']:.2f}",
         f"{lora['parameters']['optimiser_state_mb']:.2f}"),
        ("throughput (tok/s)", f"{full['tokens_per_second']:,.0f}",
         f"{lora['tokens_per_second']:,.0f}"),
        ("wall time (s)", f"{full['wall_seconds']:.0f}", f"{lora['wall_seconds']:.0f}"),
        ("NEW domain loss", f"{full['final_new_domain_loss']:.3f}",
         f"{lora['final_new_domain_loss']:.3f}"),
        ("ORIGINAL domain loss", f"{full['final_original_domain_loss']:.3f}",
         f"{lora['final_original_domain_loss']:.3f}"),
    ]
    for label, a, b in rows:
        print(f"{label:34s} {a:>16s} {b:>16s}")

    print("\n" + "-" * 88)
    print("adaptation (how much the new domain improved):")
    print(f"  full  {before['new_domain_loss']:.3f} -> {full['final_new_domain_loss']:.3f}  "
          f"({full['final_new_domain_loss'] - before['new_domain_loss']:+.3f})")
    print(f"  lora  {before['new_domain_loss']:.3f} -> {lora['final_new_domain_loss']:.3f}  "
          f"({lora['final_new_domain_loss'] - before['new_domain_loss']:+.3f})")

    print("\nforgetting (how much the original domain degraded - lower is better):")
    full_forget = full["final_original_domain_loss"] - before["original_domain_loss"]
    lora_forget = lora["final_original_domain_loss"] - before["original_domain_loss"]
    print(f"  full  {before['original_domain_loss']:.3f} -> "
          f"{full['final_original_domain_loss']:.3f}  ({full_forget:+.3f})")
    print(f"  lora  {before['original_domain_loss']:.3f} -> "
          f"{lora['final_original_domain_loss']:.3f}  ({lora_forget:+.3f})")

    if full_forget > lora_forget:
        print(f"\n  LoRA forgot {full_forget - lora_forget:.3f} less of the original domain.")
    else:
        print(f"\n  Full fine-tuning forgot {lora_forget - full_forget:.3f} less - "
              f"unexpected, and worth explaining rather than burying.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", action="store_true",
                        help="train and save the base model first")
    args = parser.parse_args()
    if args.pretrain:
        pretrain()
    else:
        compare()


if __name__ == "__main__":
    main()
