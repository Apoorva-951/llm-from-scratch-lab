"""Run the ablation suite.

Resumable: each completed run writes its own JSON, and a run whose file already
exists is skipped. A suite of a dozen runs takes long enough that losing it to
one crash would matter.

    python experiments/run.py --suite ablations
    python experiments/run.py --suite vocab
    python experiments/run.py --only baseline heads-1 --force
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ablations import build_suite, build_vocab_suite  # noqa: E402

from lm.train import train_experiment  # noqa: E402

RESULTS = ROOT / "results" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["ablations", "vocab", "all"], default="ablations")
    parser.add_argument("--only", nargs="*", help="run only these experiment names")
    parser.add_argument("--force", action="store_true", help="re-run even if results exist")
    parser.add_argument("--tokens", type=int, help="override the token budget")
    args = parser.parse_args()

    if args.suite == "ablations":
        suite = build_suite()
    elif args.suite == "vocab":
        suite = build_vocab_suite()
    else:
        suite = build_suite() + build_vocab_suite()

    if args.only:
        wanted = set(args.only)
        suite = [c for c in suite if c.name in wanted]
        missing = wanted - {c.name for c in suite}
        if missing:
            print(f"unknown experiment names: {sorted(missing)}")
            return

    if args.tokens:
        for cfg in suite:
            cfg.train.train_tokens = args.tokens

    RESULTS.mkdir(parents=True, exist_ok=True)

    pending = []
    for cfg in suite:
        path = RESULTS / f"{cfg.name}.json"
        if path.exists() and not args.force:
            print(f"skip  {cfg.name} (already run)")
            continue
        pending.append((cfg, path))

    print(f"\n{len(pending)} run(s) to go\n" + "=" * 72)

    failures: list[tuple[str, str]] = []
    for index, (cfg, path) in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] {cfg.name}")
        print(f"  hypothesis: {cfg.hypothesis}")
        try:
            record = train_experiment(cfg)
            record.save(path)
            print(f"  wrote {path.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001 - one failure must not end the suite
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            failures.append((cfg.name, str(exc)))

    print("\n" + "=" * 72)
    print(f"completed {len(pending) - len(failures)}/{len(pending)}")
    for name, error in failures:
        print(f"  FAILED {name}: {error}")


if __name__ == "__main__":
    main()
