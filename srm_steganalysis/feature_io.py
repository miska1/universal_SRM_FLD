from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureMatrix:
    X: np.ndarray
    image_names: tuple[str, ...]
    submodel_names: tuple[str, ...]
    submodel_sizes: tuple[int, ...]

    @property
    def schema_hash(self) -> str:
        return feature_schema_hash(self.submodel_names, self.submodel_sizes)


def _portable_basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def feature_schema_hash(names: Sequence[str], sizes: Sequence[int]) -> str:
    descriptor = "\n".join(f"{name}:{size}" for name, size in zip(names, sizes))
    return sha256(descriptor.encode("utf-8")).hexdigest()


def parse_feature_directory(
    directory: str | Path,
    expected_names: Sequence[str] | None = None,
) -> FeatureMatrix:
    """Read and concatenate all SRM ``.fea`` submodels.

    Rows are aligned by staged image name rather than relying on filesystem
    iteration order. Submodels are concatenated in lexical filename order, so
    the schema is reproducible across platforms.
    """
    directory = Path(directory)
    feature_files = sorted(directory.glob("*.fea"), key=lambda p: p.name)
    if not feature_files:
        raise ValueError(f"No .fea files found in {directory}")

    order = tuple(expected_names) if expected_names is not None else None
    blocks: list[np.ndarray] = []
    submodel_names: list[str] = []
    submodel_sizes: list[int] = []

    for feature_file in feature_files:
        rows: dict[str, np.ndarray] = {}
        with feature_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                tokens = line.split()
                if len(tokens) < 2:
                    raise ValueError(f"Malformed row {line_number} in {feature_file}")
                name = _portable_basename(tokens[-1])
                if name in rows:
                    raise ValueError(f"Duplicate image name {name!r} in {feature_file}")
                try:
                    rows[name] = np.asarray(tokens[:-1], dtype=np.float32)
                except ValueError as error:
                    raise ValueError(
                        f"Non-numeric feature at row {line_number} in {feature_file}"
                    ) from error

        if order is None:
            order = tuple(sorted(rows))
        missing = sorted(set(order) - set(rows))
        extra = sorted(set(rows) - set(order))
        if missing or extra:
            raise ValueError(
                f"Image rows differ in {feature_file}; missing={missing[:3]}, extra={extra[:3]}"
            )
        widths = {rows[name].size for name in order}
        if len(widths) != 1:
            raise ValueError(f"Inconsistent row widths in {feature_file}")
        block = np.stack([rows[name] for name in order]).astype(np.float32, copy=False)
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite feature in {feature_file}")
        blocks.append(block)
        submodel_names.append(feature_file.stem)
        submodel_sizes.append(block.shape[1])

    assert order is not None
    return FeatureMatrix(
        X=np.concatenate(blocks, axis=1),
        image_names=order,
        submodel_names=tuple(submodel_names),
        submodel_sizes=tuple(submodel_sizes),
    )


def save_feature_dataset(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    files: Iterable[str],
    submodel_names: Sequence[str],
    submodel_sizes: Sequence[int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(submodel_names)
    sizes = tuple(int(value) for value in submodel_sizes)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.int8),
        files=np.asarray(tuple(files), dtype=np.str_),
        submodel_names=np.asarray(names, dtype=np.str_),
        submodel_sizes=np.asarray(sizes, dtype=np.int32),
        schema_hash=np.asarray(feature_schema_hash(names, sizes), dtype=np.str_),
    )


def load_feature_dataset(path: str | Path) -> dict[str, np.ndarray | str]:
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"X", "y", "files", "submodel_names", "submodel_sizes", "schema_hash"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Feature dataset is missing keys: {sorted(missing)}")
        result: dict[str, np.ndarray | str] = {
            "X": data["X"].astype(np.float32, copy=False),
            "y": data["y"].astype(np.int8, copy=False),
            "files": data["files"].astype(np.str_),
            "submodel_names": data["submodel_names"].astype(np.str_),
            "submodel_sizes": data["submodel_sizes"].astype(np.int32),
            "schema_hash": str(data["schema_hash"].item()),
        }
    X = result["X"]
    y = result["y"]
    files = result["files"]
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert isinstance(files, np.ndarray)
    if X.ndim != 2 or y.ndim != 1 or len(y) != len(X) or len(files) != len(X):
        raise ValueError(f"Invalid shapes in feature dataset {path}")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values in feature dataset {path}")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must be 0 (cover) or 1 (stego)")
    return result
