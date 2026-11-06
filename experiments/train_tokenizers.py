"""Experiment 1: how vocabulary size trades against model size.

At small scale the embedding table dominates parameter count. With GPT-2's
50,257-token vocabulary and emb_dim=384, token embeddings plus an untied output
head cost ~38.6M parameters - roughly 78% of a 49M model, leaving only ~10M for
the transformer itself. Training a domain-specific vocabulary is therefore not a
detail; it decides how much of the parameter budget does any actual modelling.

This script trains BPE at several vocabulary sizes and measures the trade:
compression (characters per token, which sets sequence length and therefore
compute) against embedding cost.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lm.tokenizer import BPETokenizer  # noqa: E402

VOCAB_SIZES = [2048, 8192, 16384]
TRAIN_CHARS = 5_000_000
EMB_DIM = 384  # reference dimension for the parameter-cost column

SAMPLE = (
    "Once upon a time, there was a little girl named Lily. She saw a shiny red "
    "ball in the park and wanted to play with it."
)


def main() -> None:
    corpus = (ROOT / "data/tinystories-train.txt").read_text(encoding="utf-8", errors="ignore")
    text = corpus[:TRAIN_CHARS]
    held_out = corpus[TRAIN_CHARS : TRAIN_CHARS + 500_000]
    print(f"training on {len(text)/1e6:.1f}M chars, measuring on {len(held_out)/1e6:.1f}M held-out\n")

    rows = []
    for vocab_size in VOCAB_SIZES:
        started = time.time()
        tokenizer = BPETokenizer.train(text, vocab_size=vocab_size)
        train_seconds = time.time() - started

        out_path = ROOT / f"data/tokenizer-{vocab_size}.json"
        tokenizer.save(out_path)

        # Compression is measured on held-out text: a vocabulary trained on the
        # same text it is scored on would flatter larger vocabularies.
        encoded = tokenizer.encode(held_out)
        chars_per_token = len(held_out) / len(encoded)

        # Untied embeddings: one table in, one head out.
        embedding_params = 2 * len(tokenizer) * EMB_DIM

        rows.append(
            {
                "vocab_size": len(tokenizer),
                "merges": len(tokenizer.merges),
                "chars_per_token": round(chars_per_token, 3),
                "tokens_per_1k_chars": round(1000 / chars_per_token, 1),
                "embedding_params_at_384": embedding_params,
                "train_seconds": round(train_seconds, 1),
                "sample_token_count": len(tokenizer.encode(SAMPLE)),
                "roundtrip_exact": tokenizer.decode(tokenizer.encode(held_out[:20000]))
                == held_out[:20000],
            }
        )
        print(
            f"vocab {len(tokenizer):6,}  "
            f"{chars_per_token:5.2f} chars/token  "
            f"emb params {embedding_params/1e6:5.1f}M  "
            f"trained in {train_seconds:5.1f}s  "
            f"roundtrip={rows[-1]['roundtrip_exact']}",
            flush=True,
        )

    results_path = ROOT / "results/tokenizer_study.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"emb_dim": EMB_DIM, "rows": rows}, indent=2))

    print(f"\nwrote {results_path}")
    print("\nFor reference, GPT-2's vocabulary at the same embedding dimension:")
    print(f"  vocab 50,257 -> embedding params {2*50257*EMB_DIM/1e6:.1f}M")


if __name__ == "__main__":
    main()
