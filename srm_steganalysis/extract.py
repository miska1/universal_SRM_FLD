from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .feature_io import FeatureMatrix, parse_feature_directory, save_feature_dataset
from .logging_utils import LOGGER_NAME, elapsed_text


SUPPORTED_SUFFIXES = {".pgm", ".png", ".bmp", ".tif", ".tiff"}
LOGGER = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class ExtractionConfig:
    threshold: int = 2
    order: int = 4
    erase_lsb: bool = False
    parity: bool = False
    batch_size: int = 100
    workers: int = 1
    verbose: bool = False


@dataclass(frozen=True)
class ExtractorRun:
    elapsed: float
    stdout: str
    stderr: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class BatchResult:
    batch_index: int
    start: int
    features: FeatureMatrix
    extractor_elapsed: float
    parsing_elapsed: float
    total_elapsed: float
    worker_pid: int
    stdout: str
    stderr: str
    command: tuple[str, ...]


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
) -> ExtractorRun:
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
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        platform_hint = (
            " The supplied SRM.exe runs on Windows; on Linux/macOS build the portable "
            "extractor described in README.md."
        )
        raise RuntimeError(f"Could not start SRM extractor {executable}.{platform_hint}") from error
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        details = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        raise RuntimeError(f"SRM extractor failed with code {completed.returncode}:\n{details}")
    return ExtractorRun(
        elapsed=elapsed,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        command=tuple(command),
    )


def _extract_batch(
    batch_index: int,
    start: int,
    batch: tuple[Path, ...],
    executable: Path,
    config: ExtractionConfig,
) -> BatchResult:
    """Extract one independent batch; safe to call in a worker process."""
    batch_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"srm_batch_{batch_index:04d}_") as temporary:
        temporary_path = Path(temporary)
        input_dir = temporary_path / "input"
        output_dir = temporary_path / "features"
        input_dir.mkdir()
        output_dir.mkdir()
        staged_names: list[str] = []
        for offset, source in enumerate(batch):
            staged_name = f"img_{start + offset:08d}.pgm"
            _stage_as_pgm(source, input_dir / staged_name)
            staged_names.append(staged_name)

        extractor_run = _run_extractor(executable, input_dir, output_dir, config)
        parsing_started = time.perf_counter()
        parsed = parse_feature_directory(output_dir, expected_names=staged_names)
        parsing_elapsed = time.perf_counter() - parsing_started
        features = FeatureMatrix(
            X=parsed.X,
            image_names=tuple(str(path) for path in batch),
            submodel_names=parsed.submodel_names,
            submodel_sizes=parsed.submodel_sizes,
        )
    return BatchResult(
        batch_index=batch_index,
        start=start,
        features=features,
        extractor_elapsed=extractor_run.elapsed,
        parsing_elapsed=parsing_elapsed,
        total_elapsed=time.perf_counter() - batch_started,
        worker_pid=os.getpid(),
        stdout=extractor_run.stdout,
        stderr=extractor_run.stderr,
        command=extractor_run.command,
    )


def _log_completed_batch(result: BatchResult, completed: int, batch_count: int) -> None:
    LOGGER.debug("Batch %d SRM command: %s", result.batch_index, " ".join(result.command))
    if result.stdout:
        LOGGER.debug("Batch %d SRM stdout:\n%s", result.batch_index, result.stdout)
    if result.stderr:
        LOGGER.debug("Batch %d SRM stderr:\n%s", result.batch_index, result.stderr)
    LOGGER.info(
        "Completed batch %d/%d (%d of %d finished) by process %d in %s "
        "(SRM %s, parsing %s)",
        result.batch_index,
        batch_count,
        completed,
        batch_count,
        result.worker_pid,
        elapsed_text(result.total_elapsed),
        elapsed_text(result.extractor_elapsed),
        elapsed_text(result.parsing_elapsed),
    )


def extract_images(
    images: Sequence[Path],
    executable: str | Path,
    config: ExtractionConfig,
) -> FeatureMatrix:
    extraction_started = time.perf_counter()
    if not images:
        raise ValueError("No images supplied")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.workers < 1:
        raise ValueError("workers must be positive")
    executable = Path(executable)
    if not executable.is_file():
        raise FileNotFoundError(f"SRM extractor not found: {executable}")

    batch_count = (len(images) + config.batch_size - 1) // config.batch_size
    active_workers = min(config.workers, batch_count)
    LOGGER.info(
        "Starting SRM extraction: %d image(s), %d batch(es), batch size %d, process(es) %d",
        len(images),
        batch_count,
        config.batch_size,
        active_workers,
    )
    jobs = [
        (
            batch_index,
            start,
            tuple(Path(path) for path in images[start : start + config.batch_size]),
        )
        for batch_index, start in enumerate(
            range(0, len(images), config.batch_size), start=1
        )
    ]
    ordered_results: list[BatchResult | None] = [None] * batch_count
    if active_workers == 1:
        for completed, (batch_index, start, batch) in enumerate(jobs, start=1):
            LOGGER.info("Extracting batch %d/%d (%d image(s))", batch_index, batch_count, len(batch))
            result = _extract_batch(batch_index, start, batch, executable, config)
            ordered_results[batch_index - 1] = result
            _log_completed_batch(result, completed, batch_count)
    else:
        LOGGER.info("Launching %d parallel SRM worker processes", active_workers)
        with ProcessPoolExecutor(max_workers=active_workers) as executor:
            future_to_batch = {
                executor.submit(_extract_batch, batch_index, start, batch, executable, config): batch_index
                for batch_index, start, batch in jobs
            }
            LOGGER.info("Submitted %d extraction batch(es)", len(future_to_batch))
            for completed, future in enumerate(as_completed(future_to_batch), start=1):
                batch_index = future_to_batch[future]
                try:
                    result = future.result()
                except Exception as error:
                    for pending in future_to_batch:
                        pending.cancel()
                    raise RuntimeError(f"SRM extraction failed in batch {batch_index}: {error}") from error
                ordered_results[result.batch_index - 1] = result
                _log_completed_batch(result, completed, batch_count)

    results = [result for result in ordered_results if result is not None]
    if len(results) != batch_count:
        raise RuntimeError("One or more extraction batches did not return a result")
    schema_names = results[0].features.submodel_names
    schema_sizes = results[0].features.submodel_sizes
    for result in results[1:]:
        if (result.features.submodel_names, result.features.submodel_sizes) != (
            schema_names,
            schema_sizes,
        ):
            raise ValueError("SRM feature schema changed between extraction batches")
    result = FeatureMatrix(
        X=np.concatenate([batch.features.X for batch in results], axis=0),
        image_names=tuple(
            image_name for batch in results for image_name in batch.features.image_names
        ),
        submodel_names=schema_names,
        submodel_sizes=schema_sizes,
    )
    LOGGER.info(
        "Finished SRM extraction: shape=%s, submodels=%d, elapsed=%s",
        result.X.shape,
        len(result.submodel_names),
        elapsed_text(time.perf_counter() - extraction_started),
    )
    return result


def extract_dataset(
    data_root: str | Path,
    output_root: str | Path,
    executable: str | Path,
    splits: Sequence[str],
    config: ExtractionConfig,
) -> list[Path]:
    dataset_started = time.perf_counter()
    data_root = Path(data_root)
    output_root = Path(output_root)
    written: list[Path] = []
    reference_schema: tuple[tuple[str, ...], tuple[int, ...]] | None = None
    LOGGER.info("Dataset root: %s", data_root.resolve())
    LOGGER.info("Feature output root: %s", output_root.resolve())
    LOGGER.info("Requested splits: %s", ", ".join(splits))

    for split in splits:
        split_started = time.perf_counter()
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        class_directories = {
            "cover": split_dir / "cover",
            "stego": split_dir / "stego",
        }
        images_by_class: dict[str, dict[str, Path]] = {}
        for class_name, class_dir in class_directories.items():
            images = find_images(class_dir)
            relative_map = {path.relative_to(class_dir).as_posix(): path for path in images}
            if len(relative_map) != len(images):
                raise ValueError(f"Duplicate relative image paths in {class_dir}")
            images_by_class[class_name] = relative_map

        cover_names = set(images_by_class["cover"])
        stego_names = set(images_by_class["stego"])
        if cover_names != stego_names:
            missing_stego = sorted(cover_names - stego_names)
            missing_cover = sorted(stego_names - cover_names)
            raise ValueError(
                f"Split {split} does not contain synchronized cover/stego pairs; "
                f"missing stego={missing_stego[:3]}, missing cover={missing_cover[:3]}"
            )
        pair_names = sorted(cover_names)
        LOGGER.info("Split %s: verified %d synchronized cover/stego pair(s)", split, len(pair_names))

        class_results: list[tuple[int, Path, list[Path], FeatureMatrix]] = []
        for label, class_name in ((0, "cover"), (1, "stego")):
            class_dir = class_directories[class_name]
            images = [images_by_class[class_name][name] for name in pair_names]
            LOGGER.info("Split %s, class %s: found %d image(s)", split, class_name, len(images))
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
        LOGGER.info(
            "Saved split %s to %s in %s",
            split,
            output_path.resolve(),
            elapsed_text(time.perf_counter() - split_started),
        )
    LOGGER.info("All extraction completed in %s", elapsed_text(time.perf_counter() - dataset_started))
    return written


def default_extractor_candidates(project_root: Path) -> list[Path]:
    executable_name = "srm_extract.exe" if os.name == "nt" else "srm_extract"
    candidates = [
        project_root / "build" / "srm" / executable_name,
        project_root / "build" / "srm" / "Release" / executable_name,
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
