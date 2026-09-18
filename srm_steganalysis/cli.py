from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .extract import ExtractionConfig, extract_dataset, resolve_extractor
from .feature_io import load_feature_dataset
from .fld import FisherLDAEnsemble
from .logging_utils import LOGGER_NAME, configure_logging, elapsed_text
from .metrics import binary_metrics, select_balanced_threshold, write_evaluation
from .model_io import load_model, save_model


def _dataset(path: str | Path) -> tuple[dict[str, np.ndarray | str], np.ndarray, np.ndarray]:
    data = load_feature_dataset(path)
    X, y = data["X"], data["y"]
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    return data, X, y


def command_extract(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    log_file = Path(args.log_file) if args.log_file else Path(args.output) / "extraction.log"
    logger = configure_logging(log_file, args.verbose)
    logger.info("Starting feature-extraction command")
    project_root = Path(__file__).resolve().parents[1]
    executable = resolve_extractor(args.srm_exe, project_root)
    logger.info("SRM executable: %s", executable.resolve())
    config = ExtractionConfig(
        threshold=args.T,
        order=args.order,
        erase_lsb=args.erase_lsb,
        parity=args.parity,
        batch_size=args.batch_size,
        workers=args.workers,
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
        logger.info("Feature dataset ready: %s", output.resolve())
    logger.info("Feature-extraction command completed in %s", elapsed_text(time.perf_counter() - started))
    return 0


def command_train(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    log_file = Path(args.log_file) if args.log_file else Path(args.model).parent / "training.log"
    logger = configure_logging(log_file, args.verbose)
    logger.info("Starting training command")
    logger.info("Training features: %s", Path(args.train).resolve())
    if args.validation:
        logger.info("Validation features: %s", Path(args.validation).resolve())

    timing: dict[str, float] = {}
    stage_started = time.perf_counter()
    train_data, X_train, y_train = _dataset(args.train)
    timing["training_data_load"] = time.perf_counter() - stage_started
    logger.info(
        "Loaded training data: shape=%s, cover=%d, stego=%d, elapsed=%s",
        X_train.shape,
        int(np.count_nonzero(y_train == 0)),
        int(np.count_nonzero(y_train == 1)),
        elapsed_text(timing["training_data_load"]),
    )
    stage_started = time.perf_counter()
    model = FisherLDAEnsemble(
        n_estimators=args.estimators,
        subspace_size=args.subspace_size,
        shrinkage=args.shrinkage,
        ridge=args.ridge,
        bootstrap=not args.no_bootstrap,
        random_state=args.seed,
    ).fit(X_train, y_train)
    timing["model_fit"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    train_scores = model.decision_function(X_train)
    timing["training_scoring"] = time.perf_counter() - stage_started
    report: dict[str, object] = {
        "trained_estimators": len(model.learners_),
    }
    if args.validation:
        stage_started = time.perf_counter()
        validation_data, X_validation, y_validation = _dataset(args.validation)
        logger.info(
            "Loaded validation data: shape=%s, cover=%d, stego=%d",
            X_validation.shape,
            int(np.count_nonzero(y_validation == 0)),
            int(np.count_nonzero(y_validation == 1)),
        )
        if validation_data["schema_hash"] != train_data["schema_hash"]:
            raise ValueError("Training and validation feature schemas differ")
        validation_scores = model.decision_function(X_validation)
        model.set_threshold(select_balanced_threshold(y_validation, validation_scores))
        report["validation"] = binary_metrics(y_validation, validation_scores, model.threshold_)
        report["threshold_selected_on"] = "validation"
        timing["validation"] = time.perf_counter() - stage_started
        logger.info(
            "Selected validation threshold %.8f in %s",
            model.threshold_,
            elapsed_text(timing["validation"]),
        )
    else:
        report["threshold_selected_on"] = "equal-prior Fisher boundary"
        logger.info("Using equal-prior Fisher decision threshold %.8f", model.threshold_)
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
    stage_started = time.perf_counter()
    save_model(args.model, model, metadata)
    timing["model_save"] = time.perf_counter() - stage_started
    logger.info("Saved trained model: %s", Path(args.model).resolve())
    report_path = Path(args.report) if args.report else Path(args.model).with_suffix(".training.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timing["total"] = time.perf_counter() - started
    report["timing_seconds"] = {key: round(value, 6) for key, value in timing.items()}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Saved training report: %s", report_path.resolve())
    logger.info(
        "Training command completed in %s (model fitting %s)",
        elapsed_text(time.perf_counter() - started),
        elapsed_text(timing["model_fit"]),
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    log_file = Path(args.log_file) if args.log_file else Path(args.output) / "evaluation.log"
    logger = configure_logging(log_file, args.verbose)
    logger.info("Starting evaluation command")
    data, X, y = _dataset(args.features)
    logger.info("Loaded evaluation data: shape=%s", X.shape)
    model, metadata = load_model(args.model)
    if data["schema_hash"] != metadata.get("schema_hash"):
        raise ValueError("Evaluation feature schema differs from the trained model")
    scores = model.decision_function(X)
    files = np.asarray(data["files"]).astype(str).tolist()
    metrics = write_evaluation(args.output, y, scores, model.threshold_, files)
    logger.info("Evaluation outputs saved to %s", Path(args.output).resolve())
    logger.info("Evaluation command completed in %s", elapsed_text(time.perf_counter() - started))
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
    extract_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel SRM worker processes (default: 1)",
    )
    extract_parser.add_argument("--T", type=int, default=2)
    extract_parser.add_argument("--order", type=int, default=4)
    extract_parser.add_argument("--erase-lsb", action="store_true")
    extract_parser.add_argument("--parity", action="store_true")
    extract_parser.add_argument("--log-file", help="Log path (default: OUTPUT/extraction.log)")
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
    train_parser.add_argument("--log-file", help="Log path (default: MODEL_DIRECTORY/training.log)")
    train_parser.add_argument("--verbose", action="store_true")
    train_parser.set_defaults(func=command_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("--features", required=True)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--log-file", help="Log path (default: OUTPUT/evaluation.log)")
    evaluate_parser.add_argument("--verbose", action="store_true")
    evaluate_parser.set_defaults(func=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger = logging.getLogger(LOGGER_NAME)
        if logger.handlers:
            logger.error("Command failed: %s", error)
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
