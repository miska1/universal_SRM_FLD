from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


MODEL_FORMAT_VERSION = 2


def save_model(path: str | Path, model: Any, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": MODEL_FORMAT_VERSION, "model": model, "metadata": metadata}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported model file")
    return payload["model"], payload["metadata"]
