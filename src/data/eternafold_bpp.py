from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

import torch


log = logging.getLogger(__name__)

IDX_TO_RESNAME = {0: "A", 1: "C", 2: "G", 3: "U", 4: "N"}
_CACHE_VERSION = 1


def tokens_to_rna_string(tokens: torch.Tensor) -> str:
    return "".join(IDX_TO_RESNAME.get(int(tok), "N") for tok in tokens.tolist())


def resolve_eternafold_binary(binary_path: str | Path | None = None) -> Path:
    raw = str(binary_path).strip() if binary_path is not None else ""
    if not raw:
        raw = os.environ.get("ETERNAFOLD_PATH", "").strip()
    if not raw:
        raise ValueError(
            "RNA BPP features require EternaFold. Set `rna_bpp_binary_path` or the "
            "`ETERNAFOLD_PATH` environment variable."
        )

    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        for name in ("contrafold.exe", "contrafold"):
            binary = candidate / name
            if binary.exists():
                return binary.resolve()
    elif candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Could not find EternaFold contrafold binary from: {raw}")


def resolve_eternafold_parameters(
    parameters_path: str | Path | None = None,
    binary_path: str | Path | None = None,
) -> Path:
    raw = str(parameters_path).strip() if parameters_path is not None else ""
    if not raw:
        raw = os.environ.get("ETERNAFOLD_PARAMETERS", "").strip()

    if raw:
        candidate = Path(raw).expanduser()
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Could not find EternaFold parameter file from: {raw}")

    binary = resolve_eternafold_binary(binary_path=binary_path)
    candidate = binary.parent.parent / "parameters" / "EternaFoldParams.v1"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        "Could not infer EternaFold parameter file. Set `rna_bpp_parameters_path` or "
        "`ETERNAFOLD_PARAMETERS`."
    )


def resolve_eternafold_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is None or not str(cache_dir).strip():
        raise ValueError("RNA BPP features require `rna_bpp_cache_dir` to be set.")
    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_target_id(target_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(target_id))
    return safe or "target"


def _cache_payload_key(
    sequence: str,
    max_span: int,
    cutoff: float,
    binary_path: Path,
    parameters_path: Path,
) -> str:
    payload = {
        "v": _CACHE_VERSION,
        "sequence": sequence,
        "max_span": int(max_span),
        "cutoff": float(cutoff),
        "binary_path": str(binary_path).lower(),
        "parameters_path": str(parameters_path).lower(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def eternafold_cache_path(
    target_id: str,
    sequence: str,
    max_span: int,
    cutoff: float,
    binary_path: Path,
    parameters_path: Path,
    cache_dir: Path,
) -> Path:
    key = _cache_payload_key(
        sequence=sequence,
        max_span=max_span,
        cutoff=cutoff,
        binary_path=binary_path,
        parameters_path=parameters_path,
    )
    return cache_dir / f"{_sanitize_target_id(target_id)}.{key}.pt"


def _write_contrafold_input(sequence: str, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for idx, base in enumerate(sequence, start=1):
            handle.write(f"{idx}\t{base}\t-1\n")


def _parse_posterior_file(sequence: str, posterior_path: Path, max_span: int) -> torch.Tensor:
    seq_len = len(sequence)
    span = max(1, int(max_span))
    out = torch.zeros((seq_len, span), dtype=torch.float32)
    with posterior_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                i = int(fields[0]) - 1
            except ValueError:
                continue
            if not (0 <= i < seq_len):
                continue
            for item in fields[2:]:
                if ":" not in item:
                    continue
                j_token, prob_token = item.split(":", 1)
                try:
                    j = int(j_token) - 1
                    prob = float(prob_token)
                except ValueError:
                    continue
                band_idx = j - i - 1
                if 0 <= band_idx < span:
                    out[i, band_idx] = max(out[i, band_idx], prob)
    return out


def compute_eternafold_bpp_banded(
    residue_idx: torch.Tensor,
    max_span: int,
    cutoff: float,
    binary_path: str | Path | None,
    parameters_path: str | Path | None,
    scratch_dir: str | Path | None = None,
) -> torch.Tensor:
    sequence = tokens_to_rna_string(residue_idx)
    if "N" in sequence:
        raise ValueError("EternaFold only supports canonical RNA tokens A/C/G/U in the BPP path.")
    if len(sequence) <= 1:
        return torch.zeros((len(sequence), max(1, int(max_span))), dtype=torch.float32)

    binary = resolve_eternafold_binary(binary_path=binary_path)
    params = resolve_eternafold_parameters(parameters_path=parameters_path, binary_path=binary.parent)

    scratch_root = None
    if scratch_dir is not None and str(scratch_dir).strip():
        scratch_root = Path(scratch_dir).expanduser().resolve()
        scratch_root.mkdir(parents=True, exist_ok=True)

    tmp_parent = scratch_root if scratch_root is not None else Path(tempfile.gettempdir()).resolve()
    tmp_root = tmp_parent / f"eternafold_bpp_{uuid.uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=False)
    try:
        input_path = tmp_root / "query.bpseq"
        posterior_path = tmp_root / "query.posteriors"
        _write_contrafold_input(sequence=sequence, path=input_path)

        command = [
            str(binary),
            "predict",
            str(input_path),
            "--params",
            str(params),
            "--posteriors",
            str(float(cutoff)),
            str(posterior_path),
        ]
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(f"EternaFold failed for sequence length {len(sequence)}: {stderr or proc.stdout.strip()}")
        if not posterior_path.exists():
            raise RuntimeError(f"EternaFold did not produce posterior output at {posterior_path}")
        return _parse_posterior_file(sequence=sequence, posterior_path=posterior_path, max_span=max_span)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _load_cached_tensor(cache_path: Path) -> torch.Tensor:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "rna_bpp_banded" in payload:
        tensor = payload["rna_bpp_banded"]
        if isinstance(tensor, torch.Tensor):
            return tensor.to(dtype=torch.float32, device="cpu")
    raise ValueError(f"Malformed EternaFold cache payload: {cache_path}")


def _try_acquire_lock(lock_path: Path) -> bool:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError:
        return False
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
    finally:
        os.close(fd)
    return True


def _maybe_clear_stale_lock(lock_path: Path, stale_after_seconds: float) -> None:
    if not lock_path.exists():
        return
    age = time.time() - lock_path.stat().st_mtime
    if age >= stale_after_seconds:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def prune_stale_eternafold_cache_locks(
    cache_dir: str | Path,
    stale_after_seconds: float = 1800.0,
    max_examples_to_log: int = 3,
) -> int:
    cache_root = resolve_eternafold_cache_dir(cache_dir)
    removed: list[str] = []
    total_removed = 0

    for lock_path in cache_root.glob("*.lock"):
        if not lock_path.exists():
            continue
        age = time.time() - lock_path.stat().st_mtime
        if age < stale_after_seconds:
            continue
        try:
            lock_path.unlink()
            total_removed += 1
            if len(removed) < max(0, int(max_examples_to_log)):
                removed.append(str(lock_path))
        except FileNotFoundError:
            continue

    if total_removed > 0:
        suffix = ""
        if removed:
            suffix = f" Examples: {'; '.join(removed)}"
        log.warning("Removed %d stale EternaFold cache locks from %s.%s", total_removed, cache_root, suffix)

    return total_removed


def load_or_compute_eternafold_bpp_banded(
    target_id: str,
    residue_idx: torch.Tensor,
    max_span: int,
    cutoff: float,
    binary_path: str | Path | None,
    parameters_path: str | Path | None,
    cache_dir: str | Path,
    stale_lock_seconds: float = 1800.0,
    poll_interval_seconds: float = 0.1,
) -> torch.Tensor:
    binary = resolve_eternafold_binary(binary_path=binary_path)
    params = resolve_eternafold_parameters(parameters_path=parameters_path, binary_path=binary.parent)
    cache_root = resolve_eternafold_cache_dir(cache_dir)
    sequence = tokens_to_rna_string(residue_idx)
    cache_path = eternafold_cache_path(
        target_id=target_id,
        sequence=sequence,
        max_span=max_span,
        cutoff=cutoff,
        binary_path=binary,
        parameters_path=params,
        cache_dir=cache_root,
    )
    if cache_path.exists():
        return _load_cached_tensor(cache_path)

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    while True:
        if cache_path.exists():
            return _load_cached_tensor(cache_path)
        _maybe_clear_stale_lock(lock_path=lock_path, stale_after_seconds=stale_lock_seconds)
        if _try_acquire_lock(lock_path):
            try:
                tensor = compute_eternafold_bpp_banded(
                    residue_idx=residue_idx,
                    max_span=max_span,
                    cutoff=cutoff,
                    binary_path=binary,
                    parameters_path=params,
                    scratch_dir=cache_root / ".tmp",
                )
                tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
                torch.save(
                    {
                        "target_id": str(target_id),
                        "sequence": sequence,
                        "rna_bpp_banded": tensor.to(dtype=torch.float32, device="cpu"),
                        "meta": {
                            "cache_version": _CACHE_VERSION,
                            "max_span": int(max_span),
                            "cutoff": float(cutoff),
                            "binary_path": str(binary),
                            "parameters_path": str(params),
                        },
                    },
                    tmp_path,
                )
                os.replace(tmp_path, cache_path)
                return tensor
            finally:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
        time.sleep(max(0.01, float(poll_interval_seconds)))


class EternaFoldCacheWarmer:
    def __init__(
        self,
        max_workers: int,
        max_span: int,
        cutoff: float,
        binary_path: str | Path | None,
        parameters_path: str | Path | None,
        cache_dir: str | Path,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.max_span = max(1, int(max_span))
        self.cutoff = float(cutoff)
        self.binary_path = resolve_eternafold_binary(binary_path=binary_path)
        self.parameters_path = resolve_eternafold_parameters(
            parameters_path=parameters_path,
            binary_path=self.binary_path.parent,
        )
        self.cache_dir = resolve_eternafold_cache_dir(cache_dir)
        self._lock = Lock()
        self._submitted: set[str] = set()
        self._futures: list[Future[torch.Tensor]] = []
        self._progress_total = 0
        self._progress_cached = 0
        self._progress_completed = 0
        self._progress_failed = 0
        self._last_progress_log_time = 0.0
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="eternafold-cache",
        )

    def _maybe_log_progress(self, *, force: bool = False) -> None:
        with self._lock:
            total = self._progress_total
            cached = self._progress_cached
            completed = self._progress_completed
            failed = self._progress_failed
            now = time.time()
            should_log = force or (total > 0 and (now - self._last_progress_log_time) >= 15.0)
            if not should_log or total <= 0:
                return
            self._last_progress_log_time = now

        ready = cached + completed
        pending = max(0, total - ready - failed)
        log.info(
            "EternaFold cache progress: ready=%d/%d (cached=%d, computed=%d, pending=%d, failed=%d).",
            ready,
            total,
            cached,
            completed,
            pending,
            failed,
        )

    def _on_future_done(self, future: Future[torch.Tensor]) -> None:
        try:
            future.result()
        except Exception:
            with self._lock:
                self._progress_failed += 1
            log.exception("EternaFold cache warmup task failed.")
        else:
            with self._lock:
                self._progress_completed += 1
        self._maybe_log_progress(force=False)

    def submit(self, target_id: str, residue_idx: torch.Tensor) -> None:
        residue_idx_cpu = residue_idx.detach().to(device="cpu", dtype=torch.long)
        sequence = tokens_to_rna_string(residue_idx_cpu)
        cache_path = eternafold_cache_path(
            target_id=target_id,
            sequence=sequence,
            max_span=self.max_span,
            cutoff=self.cutoff,
            binary_path=self.binary_path,
            parameters_path=self.parameters_path,
            cache_dir=self.cache_dir,
        )
        with self._lock:
            key = f"{target_id}:{sequence}"
            if key in self._submitted:
                return
            self._submitted.add(key)
            self._progress_total += 1
            if cache_path.exists():
                self._progress_cached += 1
                return
            future = self._executor.submit(
                load_or_compute_eternafold_bpp_banded,
                target_id,
                residue_idx_cpu,
                self.max_span,
                self.cutoff,
                self.binary_path,
                self.parameters_path,
                self.cache_dir,
            )
            self._futures.append(future)
            future.add_done_callback(self._on_future_done)

    def submit_many(self, items: Iterable[tuple[str, torch.Tensor]]) -> None:
        for target_id, residue_idx in items:
            self.submit(target_id=target_id, residue_idx=residue_idx)
        self._maybe_log_progress(force=True)

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        self._maybe_log_progress(force=True)
