from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .feature_io import FeatureMatrix, parse_feature_directory, save_feature_dataset


SUPPORTED_SUFFIXES = {".pgm", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ExtractionConfig:
    threshold: int = 2
    order: int = 4
    erase_lsb: bool = False
    parity: bool = False
    batch_size: int = 100
    verbose: bool = False


def find_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    images = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.as_posix(),
    )
    if not images:
        raise ValueError(f"No supported grayscale images found in {directory}")
    return images


def _stage_as_pgm(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        image.load()
        if image.mode != "L":
            raise ValueError(
                f"{source} has mode {image.mode!r}; SRM requires 8-bit grayscale images. "
                "Convert color data explicitly before extraction."
            )
        image.save(destination, format="PPM")


def _run_extractor(
    executable: Path,
    input_dir: Path,
    output_dir: Path,
    config: ExtractionConfig,
) -> None:
    command = [
        str(executable.resolve()),
        "--input-dir",
        str(input_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--T",
        str(config.threshold),
        "--order",
        str(config.order),
    ]
    if config.erase_lsb:
        command.append("--eraseLSB")
    if config.parity:
        command.append("--parity")
    if config.verbose:
        command.append("--verbose")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=not config.verbose,
            text=True,
        )
    except OSError as error:
        platform_hint = (
            " The supplied SRM.exe runs on Windows; on Linux/macOS build the portable "
            "extractor described in README.md."
        )
        raise RuntimeError(f"Could not start SRM extractor {executable}.{platform_hint}") from error
    if completed.returncode != 0:
        details = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        raise RuntimeError(f"SRM extractor failed with code {completed.returncode}:\n{details}")


def extract_images(
    images: Sequence[Path],
    executable: str | Path,
    config: ExtractionConfig,
) -> FeatureMatrix:
    if not images:
        raise ValueError("No images supplied")
    executable = Path(executable)
    if not executable.is_file():
        raise FileNotFoundError(f"SRM extractor not found: {executable}")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")

    matrices: list[np.ndarray] = []
    all_names: list[str] = []
    schema_names: tuple[str, ...] | None = None
    schema_sizes: tuple[int, ...] | None = None

    for start in range(0, len(images), config.batch_size):
        batch = images[start : start + config.batch_size]
        with tempfile.TemporaryDirectory(prefix="srm_batch_") as temporary:
            temporary_path = Path(temporary)
            input_dir = temporary_path / "input"
            output_dir = temporary_path / "features"
            input_dir.mkdir()
            output_dir.mkdir()
            staged_names: list[str] = []
            for offset, source in enumerate(batch):
                staged_name = f"img_{start + offset:08d}.pgm"
                _stage_as_pgm(Path(source), input_dir / staged_name)
                staged_names.append(staged_name)

            _run_extractor(executable, input_dir, output_dir, config)
            parsed = parse_feature_directory(output_dir, expected_names=staged_names)
            if schema_names is None:
                schema_names = parsed.submodel_names
                schema_sizes = parsed.submodel_sizes
            elif (schema_names, schema_sizes) != (parsed.submodel_names, parsed.submodel_sizes):
                raise ValueError("SRM feature schema changed between extraction batches")
            matrices.append(parsed.X)
            all_names.extend(str(Path(path)) for path in batch)

    assert schema_names is not None and schema_sizes is not None
    return FeatureMatrix(
        X=np.concatenate(matrices, axis=0),
        image_names=tuple(all_names),
        submodel_names=schema_names,
        submodel_sizes=schema_sizes,
    )


def extract_dataset(
    data_root: str | Path,
    output_root: str | Path,
    executable: str | Path,
    splits: Sequence[str],
    config: ExtractionConfig,
) -> list[Path]:
    data_root = Path(data_root)
    output_root = Path(output_root)
    written: list[Path] = []
    reference_schema: tuple[tuple[str, ...], tuple[int, ...]] | None = None

    for split in splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        class_results: list[tuple[int, Path, list[Path], FeatureMatrix]] = []
        for label, class_name in ((0, "cover"), (1, "stego")):
            class_dir = split_dir / class_name
            images = find_images(class_dir)
            result = extract_images(images, executable, config)
            schema = (result.submodel_names, result.submodel_sizes)
            if reference_schema is None:
                reference_schema = schema
            elif schema != reference_schema:
                raise ValueError(f"Feature schema mismatch in {split}/{class_name}")
            class_results.append((label, class_dir, images, result))

        X = np.concatenate([item[3].X for item in class_results], axis=0)
        y = np.concatenate(
            [np.full(len(item[2]), item[0], dtype=np.int8) for item in class_results]
        )
        files = [
            f"{class_dir.name}/{path.relative_to(class_dir).as_posix()}"
            for _, class_dir, images, _ in class_results
            for path in images
        ]
        assert reference_schema is not None
        output_path = output_root / f"{split}.npz"
        save_feature_dataset(
            output_path,
            X,
            y,
            files,
            reference_schema[0],
            reference_schema[1],
        )
        written.append(output_path)
    return written


def default_extractor_candidates(project_root: Path) -> list[Path]:
    executable_name = "srm_extract.exe" if os.name == "nt" else "srm_extract"
    candidates = [
        project_root / "build" / "srm" / executable_name,
        project_root / "build" / executable_name,
    ]
    if os.name == "nt":
        candidates.append(project_root / "third_party" / "srm" / "Release64bit" / "SRM.exe")
    return candidates


def resolve_extractor(value: str | Path | None, project_root: Path) -> Path:
    if value is not None:
        return Path(value)
    for candidate in default_extractor_candidates(project_root):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No SRM extractor found. Pass --srm-exe or build the portable extractor as shown in README.md."
    )
