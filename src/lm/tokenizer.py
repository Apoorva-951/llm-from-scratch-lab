"""Byte-level byte-pair encoding, trained from scratch.

Written rather than imported for two reasons. First, the tokenizer is one of the
experiments: vocabulary size directly controls the embedding parameter budget,
which dominates model size here (GPT-2's 50,257-token vocabulary costs ~38.6M
parameters at emb_dim=384 - about 78% of a 49M model, leaving only ~10M for the
transformer itself). Second, a tokenizer is small enough to implement correctly
and understand completely.

**Byte-level, deliberately.** A first version built its alphabet from the
characters seen during training, and silently dropped anything unseen at encode
time - round-tripping held-out text failed. Operating on UTF-8 bytes makes the
alphabet the 256 possible byte values, so no input is ever out-of-vocabulary and
`decode(encode(x)) == x` holds for arbitrary text. Bytes are mapped to printable
characters (the GPT-2 trick) so merges remain plain string operations and the
vocabulary stays JSON-serialisable.

Training uses the standard word-frequency formulation: the corpus is reduced to
unique pre-tokens with counts, and merges are applied to that table rather than
to raw text, which is what makes BPE training on tens of megabytes tractable in
pure Python.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

# Split on whitespace/punctuation boundaries, keeping a leading space with the
# word (GPT-2 style) so " the" and "the" are distinct and detokenisation is exact.
_PRETOKEN = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+")

END_OF_TEXT = "<|endoftext|>"


@lru_cache(maxsize=1)
def _byte_maps() -> tuple[dict[int, str], dict[str, int]]:
    """Reversible map between the 256 byte values and printable characters.

    Bytes that are already printable map to themselves; the rest are shifted
    into an unused Unicode range. This keeps every token a normal string, so
    merges are string concatenations and the vocabulary serialises to JSON.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    byte_to_char: dict[int, str] = {b: chr(b) for b in printable}
    spare = 0
    for b in range(256):
        if b not in byte_to_char:
            byte_to_char[b] = chr(256 + spare)
            spare += 1
    char_to_byte = {c: b for b, c in byte_to_char.items()}
    return byte_to_char, char_to_byte


def _text_to_symbols(text: str) -> str:
    byte_to_char, _ = _byte_maps()
    return "".join(byte_to_char[b] for b in text.encode("utf-8"))


def _symbols_to_text(symbols: str) -> str:
    _, char_to_byte = _byte_maps()
    raw = bytes(char_to_byte[c] for c in symbols if c in char_to_byte)
    # `replace` guards against a token boundary splitting a multi-byte
    # character, which can happen mid-generation.
    return raw.decode("utf-8", errors="replace")


class BPETokenizer:
    def __init__(
        self,
        merges: list[tuple[str, str]] | None = None,
        vocab: dict[str, int] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.merges: list[tuple[str, str]] = merges or []
        self.vocab: dict[str, int] = vocab or {}
        self.special_tokens = special_tokens or [END_OF_TEXT]
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}
        # Merge priority: earlier merges were more frequent and bind first.
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._encode_cache: dict[str, list[int]] = {}
        self._special_ids = {self.vocab[t] for t in self.special_tokens if t in self.vocab}

    # ------------------------------------------------------------- training --

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        progress: bool = False,
    ) -> "BPETokenizer":
        specials = special_tokens or [END_OF_TEXT]

        # Special tokens are removed before pre-tokenisation so they are never
        # split, then added to the vocabulary as atomic units.
        stripped = text
        for token in specials:
            stripped = stripped.replace(token, "\n")

        counts: Counter[str] = Counter(_PRETOKEN.findall(stripped))
        words: dict[tuple[str, ...], int] = {
            tuple(_text_to_symbols(word)): count for word, count in counts.items()
        }

        vocab: dict[str, int] = {}
        for token in specials:
            vocab[token] = len(vocab)
        # The full byte alphabet, not just the bytes observed: this is what
        # guarantees nothing is ever out-of-vocabulary at encode time.
        byte_to_char, _ = _byte_maps()
        for b in range(256):
            ch = byte_to_char[b]
            if ch not in vocab:
                vocab[ch] = len(vocab)

        merges: list[tuple[str, str]] = []
        target_merges = max(0, vocab_size - len(vocab))

        for step in range(target_merges):
            pairs: Counter[tuple[str, str]] = Counter()
            for word, count in words.items():
                for a, b in zip(word, word[1:]):
                    pairs[(a, b)] += count
            if not pairs:
                break

            best, best_count = pairs.most_common(1)[0]
            if best_count < 2:
                break  # nothing repeats; further merges would encode noise

            merged = best[0] + best[1]
            words = _apply_merge(words, best, merged)
            merges.append(best)
            vocab[merged] = len(vocab)

            if progress and (step + 1) % 1000 == 0:
                print(f"  merge {step + 1}/{target_merges}  vocab={len(vocab)}", flush=True)

        return cls(merges=merges, vocab=vocab, special_tokens=specials)

    # ------------------------------------------------------------- encoding --

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece, is_special in _split_specials(text, self.special_tokens):
            if is_special:
                ids.append(self.vocab[piece])
                continue
            for token in _PRETOKEN.findall(piece):
                ids.extend(self._encode_pretoken(token))
        return ids

    def _encode_pretoken(self, token: str) -> list[int]:
        cached = self._encode_cache.get(token)
        if cached is not None:
            return cached

        symbols = list(_text_to_symbols(token))
        while len(symbols) > 1:
            candidate = None
            candidate_rank = None
            for index, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (candidate_rank is None or rank < candidate_rank):
                    candidate, candidate_rank = index, rank
            if candidate is None:
                break
            symbols[candidate : candidate + 2] = [symbols[candidate] + symbols[candidate + 1]]

        # Every symbol is byte-derived, so all lookups succeed by construction.
        ids = [self.vocab[s] for s in symbols]
        self._encode_cache[token] = ids
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buffer: list[str] = []
        for i in ids:
            token = self.inverse_vocab.get(i)
            if token is None:
                continue
            if i in self._special_ids:
                # Flush byte buffer before emitting a literal special token.
                out.append(_symbols_to_text("".join(buffer)))
                buffer = []
                out.append(token)
            else:
                buffer.append(token)
        out.append(_symbols_to_text("".join(buffer)))
        return "".join(out)

    # ---------------------------------------------------------- persistence --

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "vocab": self.vocab,
                    "merges": [list(pair) for pair in self.merges],
                    "special_tokens": self.special_tokens,
                },
                ensure_ascii=False,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text())
        return cls(
            merges=[tuple(pair) for pair in payload["merges"]],
            vocab=payload["vocab"],
            special_tokens=payload.get("special_tokens", [END_OF_TEXT]),
        )

    @property
    def eot_id(self) -> int | None:
        return self.vocab.get(END_OF_TEXT)

    def __len__(self) -> int:
        return len(self.vocab)


def _apply_merge(
    words: dict[tuple[str, ...], int],
    pair: tuple[str, str],
    merged: str,
) -> dict[tuple[str, ...], int]:
    first, second = pair
    updated: dict[tuple[str, ...], int] = {}
    for word, count in words.items():
        if len(word) < 2:
            updated[word] = updated.get(word, 0) + count
            continue
        out: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                out.append(merged)
                i += 2
            else:
                out.append(word[i])
                i += 1
        key = tuple(out)
        updated[key] = updated.get(key, 0) + count
    return updated


def _split_specials(text: str, specials: list[str]) -> list[tuple[str, bool]]:
    """Split into (piece, is_special) segments, matching longest specials first."""
    if not specials:
        return [(text, False)]
    pattern = "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
    parts: list[tuple[str, bool]] = []
    cursor = 0
    for match in re.finditer(pattern, text):
        if match.start() > cursor:
            parts.append((text[cursor : match.start()], False))
        parts.append((match.group(0), True))
        cursor = match.end()
    if cursor < len(text):
        parts.append((text[cursor:], False))
    return parts
