#!/usr/bin/env python3
"""Install Protenix and Stanford RNA 3D Folding data in a Kaggle-style layout.

This script expects the Kaggle CLI to be available and authenticated.
Authentication can be provided through:
1) ~/.kaggle/kaggle.json
2) KAGGLE_USERNAME and KAGGLE_KEY environment variables
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


PROTENIX_DATASET = "qiweiyin/protenix-v1-adjusted"
RNA_COMPETITION = "stanford-rna-3d-folding-2"

DEFAULT_INPUT_ROOT = Path(os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"))
DEFAULT_WORKING_ROOT = Path(os.environ.get("KAGGLE_WORKING_ROOT", "/kaggle/working"))


def run_cmd(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def has_kaggle_auth() -> bool:
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_env or has_file


def ensure_kaggle_available() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install with `pip install kaggle` and ensure `kaggle` is on PATH."
        )
    if not has_kaggle_auth():
        raise RuntimeError(
            "Kaggle authentication missing. Add ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        )


def install_protenix(protenix_dest: Path, force: bool) -> None:
    protenix_dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        PROTENIX_DATASET,
        "-p",
        str(protenix_dest),
        "--unzip",
    ]
    if force:
        cmd.append("--force")
    run_cmd(cmd)


def extract_all_zips(directory: Path, overwrite: bool) -> None:
    for zip_path in sorted(directory.glob("*.zip")):
        print(f"Extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(directory)
        if overwrite:
            zip_path.unlink(missing_ok=True)


def install_rna_data(rna_dest: Path, force: bool) -> None:
    rna_dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", RNA_COMPETITION, "-p", str(rna_dest)]
    if force:
        cmd.append("--force")
    run_cmd(cmd)
    extract_all_zips(rna_dest, overwrite=force)


def verify_layout(expected_code_dir: Path, rna_dest: Path) -> None:
    required_protenix_files = [
        expected_code_dir / "checkpoint" / "protenix_base_20250630_v1.0.0.pt",
        expected_code_dir / "common" / "components.cif",
        expected_code_dir / "common" / "components.cif.rdkit_mol.pkl",
    ]
    required_rna_files = [
        rna_dest / "test_sequences.csv",
        rna_dest / "train_sequences.csv",
        rna_dest / "train_labels.csv",
        rna_dest / "validation_sequences.csv",
        rna_dest / "validation_labels.csv",
    ]

    missing = [str(p) for p in required_protenix_files + required_rna_files if not p.exists()]
    if missing:
        raise RuntimeError("Installation finished with missing files:\n" + "\n".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory for Kaggle-style inputs (default: /kaggle/input).",
    )
    parser.add_argument(
        "--working-root",
        type=Path,
        default=DEFAULT_WORKING_ROOT,
        help="Working/output directory root (default: /kaggle/working).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download for Kaggle artifacts and remove downloaded zip files after extract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    working_root = args.working_root

    protenix_dest = input_root / "datasets" / "qiweiyin" / "protenix-v1-adjusted"
    rna_dest = input_root / "stanford-rna-3d-folding-2"
    expected_code_dir = (
        protenix_dest / "Protenix-v1-adjust-v2" / "Protenix-v1-adjust-v2" / "Protenix-v1"
    )

    input_root.mkdir(parents=True, exist_ok=True)
    working_root.mkdir(parents=True, exist_ok=True)

    ensure_kaggle_available()
    install_protenix(protenix_dest=protenix_dest, force=args.force)
    install_rna_data(rna_dest=rna_dest, force=args.force)
    verify_layout(expected_code_dir=expected_code_dir, rna_dest=rna_dest)

    print("\nInstallation complete.")
    print(f"Protenix code dir: {expected_code_dir}")
    print(f"RNA data dir:      {rna_dest}")
    print(f"Working dir:       {working_root}")
    print("\nSet these env vars if needed:")
    print(f"  PROTENIX_CODE_DIR={expected_code_dir}")
    print(f"  PROTENIX_ROOT_DIR={expected_code_dir}")
    print(f"  TEST_CSV={rna_dest / 'test_sequences.csv'}")
    print(f"  SUBMISSION_CSV={working_root / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
