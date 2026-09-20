from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def select_balanced_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    errors = 0.5 * (fpr + (1.0 - tpr))
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.0
    valid_indices = np.flatnonzero(finite)
    return float(thresholds[valid_indices[np.argmin(errors[finite])]])


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    false_alarm = fp / (fp + tn) if fp + tn else float("nan")
    missed_detection = fn / (fn + tp) if fn + tp else float("nan")
    detection_probability = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    recall = detection_probability
    f1_denominator = 2 * tp + fp + fn
    f1_score = 2 * tp / f1_denominator if f1_denominator else float("nan")
    accuracy = (tp + tn) / len(y_true) if len(y_true) else float("nan")
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(1.0 - 0.5 * (false_alarm + missed_detection)),
        "false_alarm_rate": float(false_alarm),
        "missed_detection_rate": float(missed_detection),
        "detection_probability": float(detection_probability),
        "specificity": float(specificity),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "roc_auc": float(auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def write_evaluation(
    output_dir: str | Path,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    files: Sequence[str],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = binary_metrics(y_true, scores, threshold)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    predictions = (scores >= threshold).astype(np.int8)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "true_label", "predicted_label", "stego_score"])
        writer.writerows(zip(files, y_true.tolist(), predictions.tolist(), scores.tolist()))

    if len(np.unique(y_true)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        with (output_dir / "roc_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["false_positive_rate", "true_positive_rate", "threshold"])
            writer.writerows(zip(fpr.tolist(), tpr.tolist(), roc_thresholds.tolist()))
        fig, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
        axis.plot(fpr, tpr, linewidth=2, label=f"SRM + FLD (AUC={metrics['roc_auc']:.4f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.55", label="Chance")
        axis.set(xlabel="False-alarm rate", ylabel="Detection probability", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
        fig.savefig(output_dir / "roc_curve.png", dpi=180)
        plt.close(fig)
    return metrics
