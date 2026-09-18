# Validation record

Validated on Linux with Python 3 and GCC 13.3:

- the portable C++17 SRM extractor compiled without Boost;
- one PGM image from the supplied package produced exactly 106 submodels and
  34,671 finite features;
- batched dataset extraction produced the expected `(4, 34671)` matrix and
  labels `[0, 0, 1, 1]`;
- all four unit tests passed (feature parsing/alignment, dataset serialization,
  Fisher-LDA learning/model serialization, and metric definitions); and
- the complete train/evaluate CLI produced a model, training report,
  predictions, JSON metrics, CSV ROC data, and PNG ROC plot on a synthetic
  labeled dataset.
- extraction, training, and evaluation produced timestamped console/file logs;
  the training JSON report also contained per-stage and total timing values.
- two simultaneous SRM worker processes extracted independent one-image
  batches, and their results were merged back into deterministic input order.

The supplied Windows `SRM.exe` was retained but was not executed in the Linux
validation environment. It uses the same documented command-line arguments
that the Python extraction wrapper passes.
