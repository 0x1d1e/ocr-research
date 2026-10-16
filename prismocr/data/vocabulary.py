"""Versioned character vocabulary with visible unknown-character accounting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
UNKNOWN_REPLACEMENT = "\ufffd"


@dataclass(frozen=True)
class EncodedText:
    ids: tuple[int, ...]
    oov_positions: tuple[int, ...]


@dataclass(frozen=True)
class DecodedText:
    text: str
    oov_positions: tuple[int, ...]


class Vocabulary:
    """Fixed codepoint vocabulary; normalization is intentionally not implicit."""

    def __init__(self, tokens: Iterable[str], version: str = "4.0") -> None:
        self.tokens = tuple(tokens)
        if len(self.tokens) < 2 or self.tokens[:2] != (BLANK_TOKEN, UNK_TOKEN):
            raise ValueError("token IDs 0 and 1 must be blank and unknown")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")
        if any(len(token) != 1 for token in self.tokens[2:]):
            raise ValueError("non-special vocabulary entries must be one Unicode codepoint")
        self.version = version
        self._indices = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def from_texts(
        cls, texts: Iterable[str], *, max_size: int | None = 512, version: str = "4.0"
    ) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text)
        if max_size is not None and max_size < 2:
            raise ValueError("max_size must leave room for blank and unknown tokens")
        characters = sorted(counts, key=lambda char: ord(char))
        if max_size is not None and len(characters) > max_size - 2:
            # Keep the highest-coverage entries deterministically. Every omitted
            # character maps to UNK and is reported by encode/OOV metrics.
            characters = sorted(counts, key=lambda char: (-counts[char], ord(char)))[: max_size - 2]
            characters.sort(key=ord)
        return cls((BLANK_TOKEN, UNK_TOKEN, *characters), version=version)

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> EncodedText:
        ids: list[int] = []
        oov: list[int] = []
        for index, char in enumerate(text):
            token_id = self._indices.get(char)
            if token_id is None:
                token_id = self.unk_id
                oov.append(index)
            ids.append(token_id)
        return EncodedText(tuple(ids), tuple(oov))

    @staticmethod
    def min_ctc_frames(token_ids: Iterable[int]) -> int:
        values = tuple(int(i) for i in token_ids)
        return len(values) + sum(values[i] == values[i - 1] for i in range(1, len(values)))

    def decode_ctc(self, token_ids: Iterable[int]) -> DecodedText:
        output: list[str] = []
        oov_positions: list[int] = []
        previous: int | None = None
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id < 0 or token_id >= len(self.tokens):
                token_id = self.unk_id
            if token_id != self.blank_id and token_id != previous:
                if token_id == self.unk_id:
                    oov_positions.append(len(output))
                    output.append(UNKNOWN_REPLACEMENT)
                else:
                    output.append(self.tokens[token_id])
            previous = token_id
        return DecodedText("".join(output), tuple(oov_positions))

    def _payload(self) -> dict[str, object]:
        return {"schema_version": "4.0", "vocabulary_version": self.version, "tokens": list(self.tokens)}

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self._payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path: str | Path) -> None:
        payload = self._payload()
        payload["sha256"] = self.sha256
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "4.0":
            raise ValueError("unsupported vocabulary schema version")
        vocab = cls(payload["tokens"], version=payload.get("vocabulary_version", "4.0"))
        if payload.get("sha256") != vocab.sha256:
            raise ValueError("vocabulary SHA-256 does not match its contents")
        return vocab
