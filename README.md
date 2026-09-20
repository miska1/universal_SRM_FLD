# Universal image steganalysis: SRM + Fisher LDA

This project implements a conventional universal spatial-image steganalysis
pipeline:

1. extract the 34,671-dimensional Spatial Rich Model (SRM) from each image;
2. standardize each feature using training-set statistics;
3. train an ensemble of regularized Fisher linear discriminants on random
   feature subspaces and balanced bootstrap samples;
4. average the normalized discriminant scores; and
5. report accuracy, balanced accuracy, specificity, recall, F1 score,
   false-alarm rate, missed-detection rate, detection probability, ROC AUC,
   predictions, and an ROC curve.

The feature code is the supplied 2012 implementation by Fridrich and
Kodovský. Its default configuration contains 106 submodels and 34,671
features (`T=2`, fourth-order co-occurrences, and all standard symmetries).

## Why an ensemble of Fisher discriminants?

A single full-covariance Fisher LDA would require estimating and solving a
34,671 by 34,671 covariance matrix. That is statistically singular for most
steganalysis datasets and computationally wasteful. This implementation uses
random subspaces. For base learner `b`, it computes

`w_b = (S_w,b + regularization)^(-1) (mu_stego,b - mu_cover,b)`

and uses the equal-prior midpoint between the two projected means as its
decision boundary. Averaging many normalized base-learner scores preserves the
Fisher-LDA classifier requested while making SRM classification practical.

## Input layout

Use 8-bit grayscale, lossless images. PGM, PNG, BMP, and TIFF are accepted;
every input is staged losslessly as binary PGM before SRM extraction. RGB and
16-bit images are rejected instead of being silently converted.

```text
data/
  train/
    cover/
    stego/
  val/
    cover/
    stego/
  test/
    cover/
    stego/
```

Cover/stego pairs should have matching relative filenames and must stay in the
same split. Do not randomly split individual images after combining paired
cover and stego data; that leaks image content across train and test.

## 1. Install Python dependencies

From this project directory:

```bash
python -m pip install -e .
```

## 2. Build or select the SRM extractor

### Linux or macOS

The original feature implementation has a portable C++17 front end in this
project:

```bash
cmake -S third_party/srm -B build/srm -DCMAKE_BUILD_TYPE=Release
cmake --build build/srm --config Release -j
```

If CMake is unavailable but a C++17 compiler is installed, use:

```bash
./scripts/build_extractor.sh
```

### Windows

The supplied 64-bit `third_party/srm/Release64bit/SRM.exe` is ready to use.
You can alternatively build the portable front end with CMake and Visual
Studio. Pass the chosen executable explicitly with `--srm-exe` if automatic
detection does not find it.

## 3. Extract features

```bash
python -m srm_steganalysis extract-dataset \
  --data data \
  --output features \
  --splits train val test \
  --srm-exe build/srm/srm_extract \
  --batch-size 100 \
  --workers 4
```

Extraction is logged to both the console and `features/extraction.log`. The log
records discovered images, the current split/class/batch, SRM execution time,
feature-parsing time, split time, and total extraction time. Use `--verbose`
for image staging and native-extractor output, or `--log-file PATH` to choose a
different log file.

`--workers N` selects the number of independent SRM extraction processes.
Every process receives a separate batch and temporary directory; completed
batches are reordered before the final feature matrix is saved, so changing
the worker count does not change sample order or feature values. The default is
`--workers 1` for backward-compatible sequential extraction. Use a value such
as 2–4 initially because each process consumes CPU time and memory. Parallelism
exists between batches, so there must be at least as many batches as workers;
reduce `--batch-size` when a class contains too few batches.

On Windows, use:

```powershell
python -m srm_steganalysis extract-dataset `
  --data data `
  --output features `
  --splits train val test `
  --srm-exe third_party\srm\Release64bit\SRM.exe `
  --batch-size 100 `
  --workers 4
```

Each split becomes one compressed `.npz` file containing features, labels,
relative filenames, SRM submodel metadata, and a schema hash. The batch size
controls peak extraction memory and parallel task granularity; it does not
change features. The extraction log records the process ID that completed each
batch.

## 4. Train

```bash
python -m srm_steganalysis train \
  --train features/train.npz \
  --validation features/val.npz \
  --model results/srm_fld.pkl \
  --estimators 100 \
  --subspace-size 200 \
  --shrinkage 0.1 \
  --seed 12345
```

Training is logged to the console and `results/training.log`. It records the
feature-matrix dimensions, class counts, classifier settings, progress through
the Fisher learners, elapsed fitting time, threshold selection, saved files,
and total training time. The same timings are also stored under
`timing_seconds` in `srm_fld.training.json`. Use `--verbose` or
`--log-file PATH` in the same way as for extraction.

When validation data is supplied, the decision threshold minimizes balanced
validation error, `(false-alarm rate + missed-detection rate) / 2`. Without a
validation set, the equal-prior Fisher boundary is used. Never tune the
threshold on the test set.

For a stronger but slower experiment, try 200 learners and a 500-feature
subspace. Select these hyperparameters only on validation data.

## 5. Evaluate

```bash
python -m srm_steganalysis evaluate \
  --features features/test.npz \
  --model results/srm_fld.pkl \
  --output results/test
```

The output directory contains:

- `metrics.json`
- `predictions.csv`
- `roc_curve.csv`
- `roc_curve.png`
- `evaluation.log`

All log files use the format:

```text
2026-09-18 09:15:42 | INFO     | Trained Fisher learner 50/100 (50%), elapsed=8.2 s
```

Log files are appended, so repeated experiments retain their previous run
history.

Labels are fixed as `0=cover` and `1=stego`. Therefore:

- false-alarm rate = cover images classified as stego / all cover images;
- missed-detection rate = stego images classified as cover / all stego images;
- detection probability = stego images correctly detected / all stego images.
- specificity = cover images correctly classified / all cover images;
- recall = stego images correctly detected / all stego images (therefore it is
  equal to detection probability in this binary task);
- F1 score = harmonic mean of stego precision and stego recall.

## Experimental protocol

- Keep acquisition source, development pipeline, image size, and file format
  controlled and report them. SRM is content- and source-sensitive.
- Split by source image or camera, not by derivative file. All versions of the
  same scene belong in exactly one split.
- Preserve class balance for the standard average-error criterion, or report
  balanced accuracy when prevalence differs.
- Evaluate cover-source mismatch explicitly: train on one source and test on
  another as a separate robustness experiment.
- Train separate models when preprocessing differs materially. Resizing,
  demosaicing, denoising, and JPEG decompression can dominate the embedding
  signal.

## Scope and limitations

This is a spatial, grayscale detector. It is appropriate for lossless-domain
embedding methods such as WOW, S-UNIWARD, and HUGO when training examples are
generated under matched conditions. It is not a JPEG-domain rich model and it
does not provide a meaningful secret-message extraction capability. “Universal”
means that features are not designed for one embedding algorithm; it does not
mean reliable detection under arbitrary source mismatch or unseen processing.

## References

J. Fridrich and J. Kodovský, “Rich Models for Steganalysis of Digital Images,”
*IEEE Transactions on Information Forensics and Security*, 7(3), 2012.

J. Kodovský, J. Fridrich, and V. Holub, “Ensemble Classifiers for
Steganalysis of Digital Media,” *IEEE Transactions on Information Forensics
and Security*, 7(2), 2012.

See `THIRD_PARTY_NOTICE.md` for the supplied SRM implementation’s terms.
