from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

RESNAME_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
IDX_TO_RESNAME = {0: "A", 1: "C", 2: "G", 3: "U", 4: "N"}


def _tokens_to_sequence(query_tokens: torch.Tensor) -> str:
    return "".join(IDX_TO_RESNAME.get(int(tok), "N") for tok in query_tokens.tolist())


def _parse_fasta_records(path: str | Path) -> list[tuple[str, str]]:
    fasta_path = Path(path)
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header = line[1:]
                parts = []
                continue
            parts.append(line.replace(" ", "").upper().replace("T", "U"))
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def _msa_tensor_cache_root() -> Path:
    raw = os.environ.get("RNA_MSA_TENSOR_CACHE_DIR", "").strip()
    root = Path(raw).expanduser() if raw else Path(tempfile.gettempdir()) / "rna_msa_tensor_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _msa_tensor_cache_path(msa_path: Path, expected_query: str, max_rows: int) -> Path:
    stat = msa_path.stat()
    payload = {
        "path": str(msa_path.resolve()).lower(),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
        "query": expected_query,
        "max_rows": int(max_rows),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:24]
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in msa_path.stem) or "msa"
    return _msa_tensor_cache_root() / f"{safe_stem}.{key}.pt"


def _load_cached_msa_tensors(
    cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not cache_path.exists():
        return None
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    tensors = (
        payload.get("tokens"),
        payload.get("token_mask"),
        payload.get("row_valid"),
        payload.get("profile"),
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return None
    tokens, token_mask, row_valid, profile = tensors
    return (
        tokens.to(dtype=torch.long, device="cpu"),
        token_mask.to(dtype=torch.bool, device="cpu"),
        row_valid.to(dtype=torch.bool, device="cpu"),
        profile.to(dtype=torch.float32, device="cpu"),
    )


def _save_cached_msa_tensors(
    cache_path: Path,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    tokens, token_mask, row_valid, profile = tensors
    torch.save(
        {
            "tokens": tokens.to(dtype=torch.long, device="cpu"),
            "token_mask": token_mask.to(dtype=torch.bool, device="cpu"),
            "row_valid": row_valid.to(dtype=torch.bool, device="cpu"),
            "profile": profile.to(dtype=torch.float32, device="cpu"),
        },
        tmp_path,
    )
    os.replace(tmp_path, cache_path)


def build_precomputed_rna_msa_tensors(
    msa_path: str | Path,
    query_tokens: torch.Tensor,
    max_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fasta_path = Path(msa_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"Expected RNA MSA FASTA at: {fasta_path}")

    expected_query = _tokens_to_sequence(query_tokens)
    seq_len = int(query_tokens.shape[0])
    max_r = max(1, int(max_rows))
    cache_path = _msa_tensor_cache_path(fasta_path, expected_query=expected_query, max_rows=max_r)
    cached_tensors = _load_cached_msa_tensors(cache_path)
    if cached_tensors is not None:
        return cached_tensors

    records = _parse_fasta_records(fasta_path)
    if not records:
        raise ValueError(f"RNA MSA FASTA is empty: {fasta_path}")

    tokens = torch.full((max_r, seq_len), 4, dtype=torch.long)
    token_mask = torch.zeros((max_r, seq_len), dtype=torch.bool)
    row_valid = torch.zeros((max_r,), dtype=torch.bool)

    write_row = 0
    for row_idx, (_, seq) in enumerate(records):
        if len(seq) != seq_len:
            raise ValueError(
                f"RNA MSA row length mismatch for {fasta_path}: expected {seq_len}, found {len(seq)}"
            )
        ungapped = seq.replace("-", "")
        if row_idx == 0 and ungapped != expected_query:
            raise ValueError(
                f"RNA MSA query mismatch for {fasta_path}: expected '{expected_query}', found '{ungapped}'"
            )
        row_tokens = torch.tensor(
            [RESNAME_TO_IDX.get(ch, 4) if ch != "-" else 4 for ch in seq],
            dtype=torch.long,
        )
        row_mask = torch.tensor([ch != "-" for ch in seq], dtype=torch.bool)
        if not bool(row_mask.any().item()):
            continue
        tokens[write_row] = row_tokens
        token_mask[write_row] = row_mask
        row_valid[write_row] = True
        write_row += 1
        if write_row >= max_r:
            break

    if not bool(row_valid[0].item()):
        raise ValueError(f"RNA MSA FASTA did not provide a usable query row: {fasta_path}")

    profile = F.one_hot(tokens.clamp(min=0, max=4), num_classes=5).to(dtype=torch.float32)
    profile = profile * token_mask.unsqueeze(-1).float()
    denom = token_mask.sum(dim=0, keepdim=False).clamp(min=1).unsqueeze(-1).float()
    profile = profile.sum(dim=0) / denom
    tensors = (tokens, token_mask, row_valid, profile)
    try:
        _save_cached_msa_tensors(cache_path, tensors)
    except OSError:
        pass
    return tensors
