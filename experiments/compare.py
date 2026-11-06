"""Turn the raw run files into a comparison table and plots.

Twenty-one JSON files are not results; nobody reads those, including the person
who produced them. This renders them as a table grouped by what each ablation
tests, plus loss curves and two trade-off scatters.

Works on partial results, so it can be run while the suite is still going.

    python experiments/compare.py
    python experiments/compare.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
PLOTS = ROOT / "results" / "plots"

# Grouping is by the question each run answers, not alphabetical: a table that
# puts every head-count run together is readable, one sorted by name is not.
GROUPS: list[tuple[str, list[str]]] = [
    ("Reference", ["baseline"]),
    ("Controls (expected to break)", ["control-no-causal-mask", "control-no-position"]),
    ("Attention heads", ["heads-1", "heads-2", "baseline", "heads-12"]),
    ("Depth", ["layers-2", "layers-4", "baseline", "layers-10"]),
    ("Context length", ["context-128", "baseline", "context-512"]),
    ("Activation", ["baseline", "act-relu", "act-swiglu"]),
    ("Normalisation", ["baseline", "norm-rmsnorm"]),
    ("Positional encoding", ["baseline", "pos-rope", "control-no-position"]),
    ("Parameter efficiency", ["baseline", "tied-embeddings", "gqa-2kv"]),
    ("Combined architecture", ["baseline", "llama-style"]),
    ("Vocabulary size", ["vocab-2048", "vocab-8192", "vocab-16384"]),
]


def load_runs() -> dict[str, dict]:
    if not RUNS.exists():
        return {}
    runs = {}
    for path in sorted(RUNS.glob("*.json")):
        payload = json.loads(path.read_text())
        runs[payload["name"]] = payload
    return runs


def _row(run: dict, baseline_loss: float | None) -> str:
    params = run["parameters"]
    loss = run["final_val_loss"]
    if baseline_loss is None or run["name"] == "baseline":
        delta = "   ref"
    else:
        delta = f"{loss - baseline_loss:+.3f}"
    return (
        f"  {run['name']:24s} "
        f"{loss:7.3f} {delta:>7s} "
        f"{run['final_val_perplexity']:9.2f} "
        f"{params['total']/1e6:7.2f} "
        f"{params['transformer']/1e6:8.2f} "
        f"{run['tokens_per_second']:9,.0f} "
        f"{run['wall_seconds']:7.0f}"
    )


def print_table(runs: dict[str, dict]) -> None:
    baseline = runs.get("baseline", {}).get("final_val_loss")

    header = (
        f"  {'run':24s} {'val_loss':>7s} {'delta':>7s} {'perplexity':>9s} "
        f"{'params_M':>7s} {'xformer_M':>8s} {'tok/s':>9s} {'secs':>7s}"
    )

    print("=" * 96)
    print("ABLATION RESULTS")
    print("=" * 96)

    shown: set[str] = set()
    for title, names in GROUPS:
        present = [n for n in names if n in runs]
        if not present:
            continue
        print(f"\n{title}")
        print(header)
        print("  " + "-" * 92)
        for name in present:
            print(_row(runs[name], baseline))
            shown.add(name)

    leftover = [n for n in runs if n not in shown]
    if leftover:
        print("\nOther")
        print(header)
        print("  " + "-" * 92)
        for name in sorted(leftover):
            print(_row(runs[name], baseline))

    missing = {n for _, names in GROUPS for n in names} - set(runs)
    if missing:
        print(f"\nnot yet run: {', '.join(sorted(missing))}")


def print_findings(runs: dict[str, dict]) -> None:
    """Flag the results that need explaining, rather than asserting conclusions."""
    baseline = runs.get("baseline")
    if not baseline:
        return

    print("\n" + "=" * 96)
    print("AUTOMATED OBSERVATIONS")
    print("=" * 96)
    base_loss = baseline["final_val_loss"]

    control = runs.get("control-no-causal-mask")
    if control:
        collapsed = control["final_val_perplexity"] < 2.0
        verdict = "as predicted" if collapsed else "UNEXPECTED - masking may be broken"
        print(
            f"\ncausal-mask control: perplexity {control['final_val_perplexity']:.2f} "
            f"vs baseline {baseline['final_val_perplexity']:.2f} -> {verdict}"
        )
        print("  Without the mask the model sees the token it must predict, so the")
        print("  task becomes copying. A collapse here is what validates the harness.")

    no_pos = runs.get("control-no-position")
    if no_pos:
        worse = no_pos["final_val_loss"] > base_loss
        verdict = "as predicted" if worse else "UNEXPECTED - position may not be wired in"
        print(
            f"\nno-position control: val loss {no_pos['final_val_loss']:.3f} "
            f"vs baseline {base_loss:.3f} -> {verdict}"
        )

    # Anything that beats the baseline is worth explaining rather than claiming.
    improvements = [
        (name, run["final_val_loss"] - base_loss)
        for name, run in runs.items()
        if name not in {"baseline", "control-no-causal-mask"}
        and run["final_val_loss"] < base_loss
    ]
    if improvements:
        print("\nbeat the baseline (each needs an explanation, not just a claim):")
        for name, delta in sorted(improvements, key=lambda kv: kv[1]):
            print(f"  {name:24s} {delta:+.3f}")

    # Efficiency: quality per parameter and per second.
    print("\nefficiency (lower loss and higher throughput are both better):")
    ranked = sorted(runs.values(), key=lambda r: r["final_val_loss"])
    for run in ranked[:5]:
        if run["name"] == "control-no-causal-mask":
            continue
        print(
            f"  {run['name']:24s} loss {run['final_val_loss']:6.3f}  "
            f"{run['parameters']['total']/1e6:5.2f}M params  "
            f"{run['tokens_per_second']:7,.0f} tok/s"
        )


def make_plots(runs: dict[str, dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available; skipping plots")
        return

    PLOTS.mkdir(parents=True, exist_ok=True)

    # --- loss curves, one panel per ablation group ------------------------
    panels = [(t, [n for n in names if n in runs]) for t, names in GROUPS]
    panels = [(t, n) for t, n in panels if len(n) > 1]
    if panels:
        cols = 2
        rows = math.ceil(len(panels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows), squeeze=False)
        for index, (title, names) in enumerate(panels):
            ax = axes[index // cols][index % cols]
            for name in names:
                history = runs[name]["history"]
                ax.plot(
                    [h["tokens_seen"] / 1e6 for h in history],
                    [h["val_loss"] for h in history],
                    marker="o", markersize=2.5, linewidth=1.4,
                    label=name, linestyle="--" if name == "baseline" else "-",
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("tokens seen (M)", fontsize=8)
            ax.set_ylabel("validation loss", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.25)
        for index in range(len(panels), rows * cols):
            axes[index // cols][index % cols].axis("off")
        fig.tight_layout()
        fig.savefig(PLOTS / "loss_curves.png", dpi=130)
        plt.close(fig)
        print(f"\nwrote {(PLOTS / 'loss_curves.png').relative_to(ROOT)}")

    # --- trade-off scatters ------------------------------------------------
    usable = [r for r in runs.values() if r["name"] != "control-no-causal-mask"]
    if len(usable) > 2:
        fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

        for run in usable:
            left.scatter(run["parameters"]["total"] / 1e6, run["final_val_loss"], s=42)
            left.annotate(run["name"], (run["parameters"]["total"] / 1e6, run["final_val_loss"]),
                          fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        left.set_xlabel("total parameters (M)")
        left.set_ylabel("final validation loss")
        left.set_title("Quality vs model size")
        left.grid(alpha=0.25)

        for run in usable:
            right.scatter(run["tokens_per_second"], run["final_val_loss"], s=42)
            right.annotate(run["name"], (run["tokens_per_second"], run["final_val_loss"]),
                           fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        right.set_xlabel("training throughput (tokens/sec)")
        right.set_ylabel("final validation loss")
        right.set_title("Quality vs training speed")
        right.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(PLOTS / "tradeoffs.png", dpi=130)
        plt.close(fig)
        print(f"wrote {(PLOTS / 'tradeoffs.png').relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    runs = load_runs()
    if not runs:
        print("no runs found; run experiments/run.py first")
        return

    print(f"\n{len(runs)} run(s) loaded\n")
    print_table(runs)
    print_findings(runs)
    if not args.no_plots:
        make_plots(runs)

    summary = ROOT / "results" / "summary.json"
    summary.write_text(json.dumps(
        {
            name: {
                "hypothesis": run["hypothesis"],
                "final_val_loss": run["final_val_loss"],
                "final_val_perplexity": run["final_val_perplexity"],
                "parameters": run["parameters"],
                "tokens_per_second": run["tokens_per_second"],
                "wall_seconds": run["wall_seconds"],
            }
            for name, run in runs.items()
        },
        indent=2,
    ))
    print(f"wrote {summary.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
