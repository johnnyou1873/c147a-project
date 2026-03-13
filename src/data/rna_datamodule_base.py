from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.eternafold_bpp import (
    resolve_eternafold_binary,
    resolve_eternafold_cache_dir,
    resolve_eternafold_parameters,
)
from src.data.kaggle_sequence_metadata import resolve_sequences_path


RESNAME_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
BOLTZMANN_J_PER_K = 1.380649e-23
ANGSTROM_PER_METER = 1.0e10
DEFAULT_EFFECTIVE_SPRING_CONSTANT_N_PER_M = 2.0

log = logging.getLogger(__name__)


def _thermal_sigma_angstrom(
    temperature_k: float, effective_spring_constant_n_per_m: float = DEFAULT_EFFECTIVE_SPRING_CONSTANT_N_PER_M
) -> float:
    """RMS positional fluctuation from equipartition: <x^2> = k_B T / k."""
    t = max(float(temperature_k), 0.0)
    k_eff = max(float(effective_spring_constant_n_per_m), 1e-12)
    sigma_m = math.sqrt(BOLTZMANN_J_PER_K * t / k_eff)
    return sigma_m * ANGSTROM_PER_METER


class RNADataModuleBase(LightningDataModule):
    """Minimal active base for the full-template RNA datamodule."""

    def __init__(
        self,
        data_dir: str = "data/",
        labels_file: str = "train_labels.csv",
        sequences_file: str = "",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_fraction: float = 0.9,
        val_fraction: float = 0.05,
        max_residues_per_target: int = 5120,
        max_targets: Optional[int] = None,
        template_generation_max_targets: Optional[int] = None,
        temperature_k: float = 600.0,
        apply_thermal_noise_train: bool = True,
        apply_thermal_noise_eval: bool = False,
        train_use_template_only_inputs: bool = True,
        eval_use_template_only_inputs: bool = True,
        use_template_coords: bool = True,
        template_file: str = "template_coords.pt",
        precompute_templates_if_missing: bool = True,
        template_min_percent_identity: float = 50.0,
        template_min_similarity: float = 0.0,
        template_topk_count: int = 5,
        template_force_oracle_only: bool = False,
        use_rna_msa_features: bool = True,
        rna_msa_max_rows: Optional[int] = None,
        rna_msa_fasta_dir: str = "",
        use_rna_bpp_features: bool = True,
        rna_bpp_max_span: int = 256,
        rna_bpp_cutoff: float = 1e-4,
        rna_bpp_binary_path: str = "",
        rna_bpp_parameters_path: str = "",
        rna_bpp_cache_dir: str = "eternafold_bpp_cache",
        rna_bpp_num_threads: int = 16,
        template_precompute_num_threads: int = 0,
        template_length_ratio_tolerance: float = 0.3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self._rna_dataset: Optional[Any] = None
        self.batch_size_per_device = batch_size

        self.num_chain_types = 1
        self.max_copy_number = 1
        self.thermal_noise_sigma_angstrom = _thermal_sigma_angstrom(self.hparams.temperature_k)

    def _resolved_labels_and_sequences(self) -> tuple[Path, Optional[Path]]:
        labels_path = Path(self.hparams.data_dir) / self.hparams.labels_file
        sequences_path = resolve_sequences_path(
            labels_path=labels_path,
            sequences_path=Path(self.hparams.data_dir) / self.hparams.sequences_file
            if str(self.hparams.sequences_file).strip()
            else None,
        )
        return labels_path, sequences_path

    def _template_path(self) -> Path:
        return Path(self.hparams.data_dir) / self.hparams.template_file

    def _read_template_file_payload(self, template_path: Path) -> Any:
        stat = template_path.stat()
        cache_key = (str(template_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        cached = getattr(self, "_template_file_payload_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        payload = torch.load(template_path, map_location="cpu", weights_only=False)
        self._template_file_payload_cache = (cache_key, payload)
        return payload

    def _build_collate_fn(self, add_thermal_noise: bool, use_template_only_inputs: bool) -> Any:
        raise NotImplementedError

    def _make_dataloader(
        self,
        dataset: Dataset[Any],
        shuffle: bool,
        add_thermal_noise: bool,
        use_template_only_inputs: bool,
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=self._build_collate_fn(
                add_thermal_noise=add_thermal_noise,
                use_template_only_inputs=use_template_only_inputs,
            ),
        )

    def _validate_rna_bpp_configuration(self) -> None:
        if not bool(self.hparams.use_rna_bpp_features):
            return

        cache_dir = Path(self.hparams.data_dir) / str(self.hparams.rna_bpp_cache_dir)
        binary_path = resolve_eternafold_binary(self.hparams.rna_bpp_binary_path)
        parameters_path = resolve_eternafold_parameters(
            parameters_path=self.hparams.rna_bpp_parameters_path,
            binary_path=binary_path.parent,
        )
        resolve_eternafold_cache_dir(cache_dir)
        log.info(
            "RNA BPP: using EternaFold at %s with params %s, cache=%s, warmup_threads=%d.",
            binary_path,
            parameters_path,
            cache_dir,
            int(self.hparams.rna_bpp_num_threads),
        )

    def _setup_bpp_warmup_count(self) -> int:
        if not bool(self.hparams.use_rna_bpp_features):
            return 0
        batch_size = max(1, int(self.batch_size_per_device))
        worker_factor = max(1, int(self.hparams.num_workers))
        prefetch_factor = 2 if int(self.hparams.num_workers) > 0 else 1
        return max(1, batch_size * worker_factor * prefetch_factor)

    def _warm_relevant_rna_bpp_cache(self, dataset: Any, indices: Optional[Iterable[int]] = None) -> None:
        if not bool(self.hparams.use_rna_bpp_features):
            return
        selected_indices = list(indices) if indices is not None else list(range(len(dataset)))
        if not selected_indices:
            return
        warm_count = min(len(selected_indices), self._setup_bpp_warmup_count())
        if warm_count <= 0:
            return
        dataset.warm_rna_bpp_cache(indices=selected_indices[:warm_count])
        if warm_count < len(selected_indices):
            log.info(
                "Deferred EternaFold warmup for %d/%d train targets until on-demand access.",
                len(selected_indices) - warm_count,
                len(selected_indices),
            )

    def _split_indices(self, split: Dataset) -> tuple[Dataset, list[int]]:
        if isinstance(split, Subset):
            return split.dataset, list(split.indices)
        return split, list(range(len(split)))

    def _split_target_ids(self, split: Dataset) -> set[str]:
        base, indices = self._split_indices(split)
        return {base[idx].target_id for idx in indices}

    def _validate_disjoint_splits(self) -> tuple[set[str], set[str], set[str]]:
        if self.data_train is None or self.data_val is None or self.data_test is None:
            raise RuntimeError("Datamodule splits are not initialized.")

        train_ids = self._split_target_ids(self.data_train)
        val_ids = self._split_target_ids(self.data_val)
        test_ids = self._split_target_ids(self.data_test)

        train_val = sorted(train_ids.intersection(val_ids))
        train_test = sorted(train_ids.intersection(test_ids))
        val_test = sorted(val_ids.intersection(test_ids))
        if train_val or train_test or val_test:
            raise RuntimeError(
                "Split leakage detected: overlapping targets across splits. "
                f"train_intersect_val={train_val[:5]}, "
                f"train_intersect_test={train_test[:5]}, "
                f"val_intersect_test={val_test[:5]}"
            )

        log.info(
            "Split disjointness check passed: train=%d, val=%d, test=%d targets.",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )
        return train_ids, val_ids, test_ids

    def train_dataloader(self) -> DataLoader[Any]:
        return self._make_dataloader(
            dataset=self.data_train,
            shuffle=True,
            add_thermal_noise=self.hparams.apply_thermal_noise_train,
            use_template_only_inputs=self.hparams.train_use_template_only_inputs,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return self._make_dataloader(
            dataset=self.data_val,
            shuffle=False,
            add_thermal_noise=self.hparams.apply_thermal_noise_eval,
            use_template_only_inputs=self.hparams.eval_use_template_only_inputs,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return self._make_dataloader(
            dataset=self.data_test,
            shuffle=False,
            add_thermal_noise=self.hparams.apply_thermal_noise_eval,
            use_template_only_inputs=self.hparams.eval_use_template_only_inputs,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        if self._rna_dataset is not None:
            self._rna_dataset.shutdown_background_workers(wait=False)

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
