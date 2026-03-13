from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path


_CHAIN_PREFIX_RE = re.compile(r"^Chains? ")
_AUTH_CHAIN_RE = re.compile(r"\[auth ([^\]]+)\]")
_VALID_BASES = {"A", "C", "G", "U", "T", "N"}


@dataclass(frozen=True)
class KaggleChainSegment:
    start: int
    end: int
    chain_id: str
    copy_idx: int


@dataclass(frozen=True)
class KaggleSequenceRecord:
    target_id: str
    sequence: str
    stoichiometry: str
    all_sequences: str
    segments: tuple[KaggleChainSegment, ...]


def normalize_path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))


def canonicalize_rna_sequence(sequence: str) -> str:
    cleaned = str(sequence).strip().replace(" ", "").upper()
    out: list[str] = []
    for ch in cleaned:
        if ch == "T":
            out.append("U")
        elif ch in _VALID_BASES:
            out.append(ch)
        else:
            out.append("N")
    return "".join(out)


def infer_sequences_path_from_labels(labels_path: str | Path) -> Path | None:
    path = Path(labels_path)
    if not path.name.endswith("_labels.csv"):
        return None
    candidate = path.with_name(path.name.replace("_labels.csv", "_sequences.csv"))
    return candidate if candidate.exists() else None


def resolve_sequences_path(labels_path: str | Path, sequences_path: str | Path | None = None) -> Path | None:
    if sequences_path is not None and str(sequences_path).strip():
        return Path(sequences_path)
    return infer_sequences_path_from_labels(labels_path)


def parse_stoichiometry(stoichiometry: str) -> list[tuple[str, int]]:
    raw = str(stoichiometry).strip()
    if not raw:
        return []

    out: list[tuple[str, int]] = []
    for part in raw.split(";"):
        token = part.strip()
        if not token or ":" not in token:
            continue
        chain_id, copies = token.split(":", 1)
        try:
            copy_count = int(copies.strip())
        except ValueError:
            continue
        chain_key = chain_id.strip()
        if chain_key and copy_count > 0:
            out.append((chain_key, copy_count))
    return out


def parse_all_sequences_fasta(fasta_content: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    lines = str(fasta_content).strip().splitlines()
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith(">"):
            idx += 1
            continue

        parts = line.split("|")
        chains_part = parts[1].strip() if len(parts) > 1 else ""
        auth_chain_ids: list[str] = []
        replaced = _CHAIN_PREFIX_RE.sub("", chains_part)
        for chain_token in replaced.split(","):
            token = chain_token.strip()
            if not token:
                continue
            auth_match = _AUTH_CHAIN_RE.search(token)
            if auth_match:
                auth_chain_ids.append(auth_match.group(1).strip())
            else:
                auth_chain_ids.append(token)

        seq_parts: list[str] = []
        while (idx + 1) < len(lines) and not lines[idx + 1].lstrip().startswith(">"):
            seq_parts.append(lines[idx + 1].strip())
            idx += 1

        sequence = canonicalize_rna_sequence("".join(seq_parts))
        if auth_chain_ids and sequence:
            auth_ids = tuple(auth_chain_ids)
            for auth_chain_id in auth_ids:
                result.setdefault(auth_chain_id, (sequence, auth_ids))
        idx += 1

    return result


def build_target_segments(
    sequence: str,
    stoichiometry: str,
    all_sequences: str,
) -> tuple[KaggleChainSegment, ...]:
    canonical_sequence = canonicalize_rna_sequence(sequence)
    order = parse_stoichiometry(stoichiometry)
    if not order:
        return ()

    chain_map = parse_all_sequences_fasta(all_sequences)
    if not chain_map:
        return ()

    segments: list[KaggleChainSegment] = []
    rebuilt_parts: list[str] = []
    pos = 0
    copies_seen: dict[str, int] = {}

    for chain_id, num_copies in order:
        base = chain_map.get(chain_id)
        if base is None:
            return ()
        chain_sequence = base[0]
        if not chain_sequence:
            return ()
        for _ in range(num_copies):
            copy_idx = copies_seen.get(chain_id, 0) + 1
            copies_seen[chain_id] = copy_idx
            next_pos = pos + len(chain_sequence)
            segments.append(
                KaggleChainSegment(
                    start=pos,
                    end=next_pos,
                    chain_id=chain_id,
                    copy_idx=copy_idx,
                )
            )
            rebuilt_parts.append(chain_sequence)
            pos = next_pos

    if pos != len(canonical_sequence):
        return ()
    if "".join(rebuilt_parts) != canonical_sequence:
        return ()
    return tuple(segments)


def load_kaggle_sequence_records(
    sequences_path: str | Path,
    max_targets: int | None = None,
) -> dict[str, KaggleSequenceRecord]:
    path = Path(sequences_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find sequences file: {path}")

    records: dict[str, KaggleSequenceRecord] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target_id = str(row.get("target_id", "")).strip()
            if not target_id or target_id in records:
                continue
            if max_targets is not None and len(records) >= max_targets:
                break

            sequence = canonicalize_rna_sequence(str(row.get("sequence", "")))
            stoichiometry = "" if row.get("stoichiometry") is None else str(row.get("stoichiometry"))
            all_sequences = "" if row.get("all_sequences") is None else str(row.get("all_sequences"))
            records[target_id] = KaggleSequenceRecord(
                target_id=target_id,
                sequence=sequence,
                stoichiometry=stoichiometry,
                all_sequences=all_sequences,
                segments=build_target_segments(
                    sequence=sequence,
                    stoichiometry=stoichiometry,
                    all_sequences=all_sequences,
                ),
            )
    return records
