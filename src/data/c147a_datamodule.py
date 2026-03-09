from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.precompute_templates import precompute_template_coords


RESNAME_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
BOLTZMANN_J_PER_K = 1.380649e-23
ANGSTROM_PER_METER = 1.0e10
DEFAULT_EFFECTIVE_SPRING_CONSTANT_N_PER_M = 2.0

log = logging.getLogger(__name__)


@dataclass
class RNAExample:
    target_id: str
    residue_idx: torch.Tensor  # (L,)
    chain_idx: torch.Tensor  # (L,)
    copy_idx: torch.Tensor  # (L,)
    resid_norm: torch.Tensor  # (L,)
    coords: torch.Tensor  # (L, 3)
    template_coords: torch.Tensor  # (L, 3)
    has_template: bool


class RNAIdentityDataset(Dataset[RNAExample]):
    def __init__(
        self,
        labels_path: str | Path,
        max_residues_per_target: int = 5120,
        max_targets: Optional[int] = 256,
        template_coords_by_target: Optional[dict[str, torch.Tensor]] = None,
        template_available_by_target: Optional[dict[str, bool]] = None,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.max_residues_per_target = max_residues_per_target
        self.max_targets = max_targets
        self.template_coords_by_target = template_coords_by_target or {}
        self.template_available_by_target = template_available_by_target or {}
        self.examples, self.num_chain_types, self.max_copy_number, self.num_examples_with_template = self._load_examples()

    def _load_examples(self) -> tuple[list[RNAExample], int, int, int]:
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Could not find labels file: {self.labels_path}")

        rows_by_target: dict[str, list[tuple[int, int, str, int, str, int, float, float, float]]] = {}
        chain_values: set[str] = set()
        max_copy = 0

        with self.labels_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_idx, row in enumerate(reader):
                full_id = row.get("ID", "")
                if "_" not in full_id:
                    continue
                target_id, pos_token = full_id.rsplit("_", 1)

                if target_id not in rows_by_target:
                    if self.max_targets is not None and len(rows_by_target) >= self.max_targets:
                        continue
                    rows_by_target[target_id] = []

                target_rows = rows_by_target[target_id]
                if len(target_rows) >= self.max_residues_per_target:
                    continue

                resname = str(row.get("resname", "N")).upper()
                chain = str(row.get("chain", "UNK"))
                try:
                    pos_idx = int(float(pos_token))
                    resid = int(float(row.get("resid", 0)))
                    copy = int(float(row.get("copy", 0)))
                    x = float(row.get("x_1", 0.0))
                    y = float(row.get("y_1", 0.0))
                    z = float(row.get("z_1", 0.0))
                except ValueError:
                    continue

                target_rows.append((pos_idx, line_idx, resname, resid, chain, copy, x, y, z))
                chain_values.add(chain)
                max_copy = max(max_copy, copy)

        chain_to_idx = {chain: idx for idx, chain in enumerate(sorted(chain_values))}
        examples: list[RNAExample] = []
        num_examples_with_template = 0

        for target_id, rows in rows_by_target.items():
            rows.sort(key=lambda x: (x[0], x[1]))  # ID suffix order, then original row order
            residues = [RESNAME_TO_IDX.get(r[2], 4) for r in rows]
            chains = [chain_to_idx[r[4]] for r in rows]
            copies = [max(0, r[5]) for r in rows]
            resids = [r[3] for r in rows]
            coords = [[r[6], r[7], r[8]] for r in rows]

            resid_t = torch.tensor(resids, dtype=torch.float32)
            denom = resid_t.max().clamp(min=1.0)
            resid_norm = resid_t / denom

            coords_t = torch.tensor(coords, dtype=torch.float32)
            template_coords = torch.zeros_like(coords_t)
            has_template = False
            if target_id in self.template_coords_by_target:
                tmpl = self.template_coords_by_target[target_id].detach().to(dtype=torch.float32, device="cpu")
                if tmpl.ndim == 2 and tmpl.shape[1] == 3:
                    seq_len = coords_t.shape[0]
                    if tmpl.shape[0] >= seq_len:
                        template_coords = tmpl[:seq_len].clone()
                    else:
                        template_coords[: tmpl.shape[0]] = tmpl
                    has_template = bool(self.template_available_by_target.get(target_id, True))

            if has_template:
                num_examples_with_template += 1

            examples.append(
                RNAExample(
                    target_id=target_id,
                    residue_idx=torch.tensor(residues, dtype=torch.long),
                    chain_idx=torch.tensor(chains, dtype=torch.long),
                    copy_idx=torch.tensor(copies, dtype=torch.long),
                    resid_norm=resid_norm,
                    coords=coords_t,
                    template_coords=template_coords,
                    has_template=has_template,
                )
            )

        if not examples:
            raise RuntimeError(f"No usable samples were found in {self.labels_path}")

        return examples, max(1, len(chain_to_idx)), max_copy, num_examples_with_template

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RNAExample:
        return self.examples[index]


def _thermal_sigma_angstrom(
    temperature_k: float, effective_spring_constant_n_per_m: float = DEFAULT_EFFECTIVE_SPRING_CONSTANT_N_PER_M
) -> float:
    """RMS positional fluctuation from equipartition: <x^2> = k_B T / k."""
    t = max(float(temperature_k), 0.0)
    k_eff = max(float(effective_spring_constant_n_per_m), 1e-12)
    sigma_m = math.sqrt(BOLTZMANN_J_PER_K * t / k_eff)
    return sigma_m * ANGSTROM_PER_METER


def _collate_rna_batch(
    batch: list[RNAExample], thermal_noise_sigma_angstrom: float = 0.0, add_thermal_noise: bool = False
) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(item.coords.shape[0] for item in batch)

    residue_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    chain_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    copy_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    resid = torch.zeros(batch_size, max_len, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    template_coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    template_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item.coords.shape[0]
        residue_idx[i, :seq_len] = item.residue_idx
        chain_idx[i, :seq_len] = item.chain_idx
        copy_idx[i, :seq_len] = item.copy_idx
        resid[i, :seq_len] = item.resid_norm
        coords[i, :seq_len] = item.coords
        template_coords[i, :seq_len] = item.template_coords
        mask[i, :seq_len] = True
        if item.has_template:
            template_mask[i, :seq_len] = True

    target_coords = coords.clone()

    if add_thermal_noise and thermal_noise_sigma_angstrom > 0.0:
        noise = torch.randn_like(coords) * thermal_noise_sigma_angstrom
        noise = noise * mask.unsqueeze(-1).float()
        coords = coords + noise

    return {
        "residue_idx": residue_idx,
        "chain_idx": chain_idx,
        "copy_idx": copy_idx,
        "resid": resid,
        "coords": coords,  # input coordinates
        "template_coords": template_coords,  # precomputed MSA/template coordinates
        "template_mask": template_mask,  # template availability mask
        "target_coords": target_coords,  # clean target coordinates
        "mask": mask,
    }


class C147ADataModule(LightningDataModule):
    """RNA identity-refinement datamodule using `train_labels.csv` as input and target."""

    def __init__(
        self,
        data_dir: str = "data/",
        labels_file: str = "train_labels.csv",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_fraction: float = 0.9,
        val_fraction: float = 0.05,
        max_residues_per_target: int = 5120,
        max_targets: Optional[int] = 256,
        temperature_k: float = 600.0,
        apply_thermal_noise_train: bool = True,
        apply_thermal_noise_eval: bool = False,
        use_template_coords: bool = True,
        template_file: str = "template_coords.pt",
        precompute_templates_if_missing: bool = True,
        template_min_percent_identity: float = 50.0,
        template_min_similarity: float = 0.1,
        template_max_templates: int = 8,
        template_length_ratio_tolerance: float = 0.3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

        self.num_chain_types = 1
        self.max_copy_number = 1
        self.thermal_noise_sigma_angstrom = _thermal_sigma_angstrom(self.hparams.temperature_k)

    def prepare_data(self) -> None:
        labels_path = Path(self.hparams.data_dir) / self.hparams.labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"Expected labels file at: {labels_path}")

        if self.hparams.use_template_coords:
            template_path = Path(self.hparams.data_dir) / self.hparams.template_file
            if not template_path.exists() and self.hparams.precompute_templates_if_missing:
                log.info(
                    "Template file not found at %s, precomputing (min_identity=%.1f%%, min_similarity=%.2f)...",
                    template_path,
                    self.hparams.template_min_percent_identity,
                    self.hparams.template_min_similarity,
                )
                precompute_template_coords(
                    labels_path=labels_path,
                    output_path=template_path,
                    min_percent_identity=self.hparams.template_min_percent_identity,
                    min_similarity=self.hparams.template_min_similarity,
                    max_templates=self.hparams.template_max_templates,
                    max_residues_per_target=self.hparams.max_residues_per_target,
                    max_targets=self.hparams.max_targets,
                    length_ratio_tolerance=self.hparams.template_length_ratio_tolerance,
                    exclude_self=True,
                )

    def _load_template_payload(self) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
        if not self.hparams.use_template_coords:
            return {}, {}

        template_path = Path(self.hparams.data_dir) / self.hparams.template_file
        if not template_path.exists():
            log.warning("Template file not found at %s. Continuing without template coordinates.", template_path)
            return {}, {}

        payload = torch.load(template_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "templates" in payload:
            templates = payload.get("templates", {})
            available = payload.get("available", {})
            return templates, available

        log.warning("Template file at %s has unexpected format. Continuing without templates.", template_path)
        return {}, {}

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by world size ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is None and self.data_val is None and self.data_test is None:
            labels_path = Path(self.hparams.data_dir) / self.hparams.labels_file
            template_coords_by_target, template_available_by_target = self._load_template_payload()
            dataset = RNAIdentityDataset(
                labels_path=labels_path,
                max_residues_per_target=self.hparams.max_residues_per_target,
                max_targets=self.hparams.max_targets,
                template_coords_by_target=template_coords_by_target,
                template_available_by_target=template_available_by_target,
            )
            self.num_chain_types = dataset.num_chain_types
            self.max_copy_number = dataset.max_copy_number

            n_total = len(dataset)
            n_train = max(1, int(n_total * self.hparams.train_fraction))
            n_val = max(1, int(n_total * self.hparams.val_fraction))
            n_test = max(1, n_total - n_train - n_val)
            if n_train + n_val + n_test > n_total:
                n_train = max(1, n_total - n_val - n_test)

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[n_train, n_val, n_total - n_train - n_val],
                generator=torch.Generator().manual_seed(self.hparams.seed),
            )

            log.info(
                "Thermal coordinate noise: T=%.1fK, sigma=%.3fA (train=%s, eval=%s); template examples=%d/%d",
                self.hparams.temperature_k,
                self.thermal_noise_sigma_angstrom,
                self.hparams.apply_thermal_noise_train,
                self.hparams.apply_thermal_noise_eval,
                dataset.num_examples_with_template,
                len(dataset),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=partial(
                _collate_rna_batch,
                thermal_noise_sigma_angstrom=self.thermal_noise_sigma_angstrom,
                add_thermal_noise=self.hparams.apply_thermal_noise_train,
            ),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=partial(
                _collate_rna_batch,
                thermal_noise_sigma_angstrom=self.thermal_noise_sigma_angstrom,
                add_thermal_noise=self.hparams.apply_thermal_noise_eval,
            ),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=partial(
                _collate_rna_batch,
                thermal_noise_sigma_angstrom=self.thermal_noise_sigma_angstrom,
                add_thermal_noise=self.hparams.apply_thermal_noise_eval,
            ),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = C147ADataModule()
