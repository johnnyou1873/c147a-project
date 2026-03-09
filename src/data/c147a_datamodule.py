from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


RESNAME_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


@dataclass
class RNAExample:
    residue_idx: torch.Tensor  # (L,)
    chain_idx: torch.Tensor  # (L,)
    copy_idx: torch.Tensor  # (L,)
    resid_norm: torch.Tensor  # (L,)
    coords: torch.Tensor  # (L, 3)


class RNAIdentityDataset(Dataset[RNAExample]):
    def __init__(
        self,
        labels_path: str | Path,
        max_residues_per_target: int = 256,
        max_targets: Optional[int] = 256,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.max_residues_per_target = max_residues_per_target
        self.max_targets = max_targets
        self.examples, self.num_chain_types, self.max_copy_number = self._load_examples()

    def _load_examples(self) -> tuple[list[RNAExample], int, int]:
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Could not find labels file: {self.labels_path}")

        rows_by_target: dict[str, list[tuple[int, str, int, str, int, float, float, float]]] = {}
        chain_values: set[str] = set()
        max_copy = 0

        with self.labels_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for line_idx, row in enumerate(reader):
                full_id = row.get("ID", "")
                if "_" not in full_id:
                    continue
                target_id = full_id.rsplit("_", 1)[0]

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
                    resid = int(float(row.get("resid", 0)))
                    copy = int(float(row.get("copy", 0)))
                    x = float(row.get("x_1", 0.0))
                    y = float(row.get("y_1", 0.0))
                    z = float(row.get("z_1", 0.0))
                except ValueError:
                    continue

                target_rows.append((line_idx, resname, resid, chain, copy, x, y, z))
                chain_values.add(chain)
                max_copy = max(max_copy, copy)

        chain_to_idx = {chain: idx for idx, chain in enumerate(sorted(chain_values))}
        examples: list[RNAExample] = []

        for _, rows in rows_by_target.items():
            rows.sort(key=lambda x: (x[4], x[3], x[2], x[0]))  # copy, chain, resid, original order
            residues = [RESNAME_TO_IDX.get(r[1], 4) for r in rows]
            chains = [chain_to_idx[r[3]] for r in rows]
            copies = [max(0, r[4]) for r in rows]
            resids = [r[2] for r in rows]
            coords = [[r[5], r[6], r[7]] for r in rows]

            resid_t = torch.tensor(resids, dtype=torch.float32)
            denom = resid_t.max().clamp(min=1.0)
            resid_norm = resid_t / denom

            examples.append(
                RNAExample(
                    residue_idx=torch.tensor(residues, dtype=torch.long),
                    chain_idx=torch.tensor(chains, dtype=torch.long),
                    copy_idx=torch.tensor(copies, dtype=torch.long),
                    resid_norm=resid_norm,
                    coords=torch.tensor(coords, dtype=torch.float32),
                )
            )

        if not examples:
            raise RuntimeError(f"No usable samples were found in {self.labels_path}")

        return examples, max(1, len(chain_to_idx)), max_copy

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RNAExample:
        return self.examples[index]


def _collate_rna_batch(batch: list[RNAExample]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(item.coords.shape[0] for item in batch)

    residue_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    chain_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    copy_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    resid = torch.zeros(batch_size, max_len, dtype=torch.float32)
    coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        seq_len = item.coords.shape[0]
        residue_idx[i, :seq_len] = item.residue_idx
        chain_idx[i, :seq_len] = item.chain_idx
        copy_idx[i, :seq_len] = item.copy_idx
        resid[i, :seq_len] = item.resid_norm
        coords[i, :seq_len] = item.coords
        mask[i, :seq_len] = True

    return {
        "residue_idx": residue_idx,
        "chain_idx": chain_idx,
        "copy_idx": copy_idx,
        "resid": resid,
        "coords": coords,  # input coordinates
        "target_coords": coords.clone(),  # identity target
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
        max_residues_per_target: int = 256,
        max_targets: Optional[int] = 256,
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

    def prepare_data(self) -> None:
        labels_path = Path(self.hparams.data_dir) / self.hparams.labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"Expected labels file at: {labels_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by world size ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is None and self.data_val is None and self.data_test is None:
            labels_path = Path(self.hparams.data_dir) / self.hparams.labels_file
            dataset = RNAIdentityDataset(
                labels_path=labels_path,
                max_residues_per_target=self.hparams.max_residues_per_target,
                max_targets=self.hparams.max_targets,
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

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=_collate_rna_batch,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=_collate_rna_batch,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            collate_fn=_collate_rna_batch,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = C147ADataModule()
