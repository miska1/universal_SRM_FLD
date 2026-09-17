from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .extract import ExtractionConfig, extract_dataset, resolve_extractor
from .feature_io import load_feature_dataset
from .fld import FisherLDAEnsemble
from .metrics import binary_metrics, select_balanced_threshold, write_evaluation
from .model_io import load_model, save_model


def _dataset(path: str | Path) -> tuple[dict[str, np.ndarray | str], np.ndarray, np.ndarray]:
    data = load_feature_dataset(path)
    X, y = data["X"], data["y"]
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    return data, X, y


def command_extract(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    executable = resolve_extractor(args.srm_exe, project_root)
    config = ExtractionConfig(
        threshold=args.T,
        order=args.order,
        erase_lsb=args.erase_lsb,
        parity=args.parity,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    outputs = extract_dataset(
        args.data,
        args.output,
        executable,
        args.splits,
        config,
    )
    for output in outputs:
        print(output)
    return 0


def command_train(args: argparse.Namespace) -> int:
    train_data, X_train, y_train = _dataset(args.train)
    model = FisherLDAEnsemble(
        n_estimators=args.estimators,
        subspace_size=args.subspace_size,
        shrinkage=args.shrinkage,
        ridge=args.ridge,
        bootstrap=not args.no_bootstrap,
        random_state=args.seed,
    ).fit(X_train, y_train)

    train_scores = model.decision_function(X_train)
    report: dict[str, object] = {
        "trained_estimators": len(model.learners_),
    }
    if args.validation:
        validation_data, X_validation, y_validation = _dataset(args.validation)
        if validation_data["schema_hash"] != train_data["schema_hash"]:
            raise ValueError("Training and validation feature schemas differ")
        validation_scores = model.decision_function(X_validation)
        model.set_threshold(select_balanced_threshold(y_validation, validation_scores))
        report["validation"] = binary_metrics(y_validation, validation_scores, model.threshold_)
        report["threshold_selected_on"] = "validation"
    else:
        report["threshold_selected_on"] = "equal-prior Fisher boundary"
    report["training"] = binary_metrics(y_train, train_scores, model.threshold_)

    metadata = {
        "schema_hash": train_data["schema_hash"],
        "submodel_names": np.asarray(train_data["submodel_names"]).tolist(),
        "submodel_sizes": np.asarray(train_data["submodel_sizes"]).astype(int).tolist(),
        "n_features": int(X_train.shape[1]),
        "labels": {"0": "cover", "1": "stego"},
        "hyperparameters": {
            "estimators": args.estimators,
            "subspace_size": args.subspace_size,
            "shrinkage": args.shrinkage,
            "ridge": args.ridge,
            "bootstrap": not args.no_bootstrap,
            "seed": args.seed,
        },
    }
    save_model(args.model, model, metadata)
    report_path = Path(args.report) if args.report else Path(args.model).with_suffix(".training.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.model)
    print(report_path)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    data, X, y = _dataset(args.features)
    model, metadata = load_model(args.model)
    if data["schema_hash"] != metadata.get("schema_hash"):
        raise ValueError("Evaluation feature schema differs from the trained model")
    scores = model.decision_function(X)
    files = np.asarray(data["files"]).astype(str).tolist()
    metrics = write_evaluation(args.output, y, scores, model.threshold_, files)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srm-steganalysis",
        description="Universal spatial image steganalysis with SRM and Fisher LDA.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-dataset", help="Extract SRM features")
    extract_parser.add_argument("--data", required=True, help="Root with SPLIT/cover and SPLIT/stego")
    extract_parser.add_argument("--output", required=True, help="Output directory for split .npz files")
    extract_parser.add_argument("--srm-exe", help="Path to SRM.exe or portable srm_extract")
    extract_parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    extract_parser.add_argument("--batch-size", type=int, default=100)
    extract_parser.add_argument("--T", type=int, default=2)
    extract_parser.add_argument("--order", type=int, default=4)
    extract_parser.add_argument("--erase-lsb", action="store_true")
    extract_parser.add_argument("--parity", action="store_true")
    extract_parser.add_argument("--verbose", action="store_true")
    extract_parser.set_defaults(func=command_extract)

    train_parser = subparsers.add_parser("train", help="Train Fisher-LDA ensemble")
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--validation")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--report")
    train_parser.add_argument("--estimators", type=int, default=100)
    train_parser.add_argument("--subspace-size", type=int, default=200)
    train_parser.add_argument("--shrinkage", type=float, default=0.1)
    train_parser.add_argument("--ridge", type=float, default=1e-6)
    train_parser.add_argument("--seed", type=int, default=12345)
    train_parser.add_argument("--no-bootstrap", action="store_true")
    train_parser.set_defaults(func=command_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("--features", required=True)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.set_defaults(func=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
