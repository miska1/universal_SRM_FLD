# Universal image steganalysis: SRM + Fisher LDA

This project implements a conventional universal spatial-image steganalysis
pipeline:

1. extract the 34,671-dimensional Spatial Rich Model (SRM) from each image;
2. synchronize every cover image with its corresponding stego image;
3. train the DDE Lab MATLAB-style ensemble of Fisher linear discriminants on
   random feature subspaces and paired bootstrap samples;
4. combine binary base-learner decisions by majority voting; and
5. report accuracy, balanced accuracy, specificity, recall, F1 score,
   false-alarm rate, missed-detection rate, detection probability, ROC AUC,
   predictions, and an ROC curve.

The feature code is the supplied 2012 implementation by Fridrich and
Kodovský. Its default configuration contains 106 submodels and 34,671
features (`T=2`, fourth-order co-occurrences, and all standard symmetries).

## Why an ensemble of Fisher discriminants?

A single full-covariance Fisher LDA would require estimating and solving a
34,671 by 34,671 covariance matrix. That is statistically singular for most
steganalysis datasets and computationally wasteful. This implementation follows
the original `ensemble_training.m` and `ensemble_testing.m` procedure and uses
random subspaces. For base learner `b`, it computes

`w_b = (S_w,b + regularization)^(-1) (mu_stego,b - mu_cover,b)`

using the sum of the cover and stego within-class covariance matrices and the
same small diagonal regularization used by the MATLAB code. Each base learner
selects the projected threshold that minimizes false alarms plus missed
detections on its paired bootstrap sample. Its output is one cover/stego vote.
The ensemble uses majority voting, and out-of-bag error can automatically
select the subspace dimension (`d_sub`) and number of learners (`L`).

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

Cover/stego pairs **must have matching relative filenames** and must stay in the
same split. Extraction rejects unmatched pairs because the original ensemble
bootstraps cover/stego pairs together. Do not randomly split individual images
after combining paired cover and stego data; that leaks image content across
train and test.

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
  --estimators automatic \
  --subspace-size automatic \
  --seed-subspaces 12345 \
  --seed-bootstrap 67890
```

Training is logged to the console and `results/training.log`. It records the
feature-matrix dimensions, synchronized pair count, classifier settings,
per-candidate out-of-bag (OOB) error, selected `d_sub` and `L`, fitting time,
saved files, and total training time. The same timings and the complete OOB
search history are stored in `srm_fld.training.json`. Use `--verbose` or
`--log-file PATH` in the same way as for extraction.

The defaults reproduce the MATLAB ensemble's automatic controls:

- `--subspace-size automatic` searches for `d_sub` using OOB error;
- `--estimators automatic` stops when the smoothed OOB error stabilizes;
- `--d-sub-step 200` and `--oob-tolerance 0.02` control the subspace search;
- `--l-min-length 25`, `--l-memory 50`, `--l-epsilon 0.005`, and
  `--l-kernel-size 5` control automatic `L`; and
- `--max-estimators 500` is the safety limit used by the original code.

You can use fixed values, for example `--estimators 100 --subspace-size 200`,
for a controlled experiment. Paired bagging is always enabled, as in MATLAB
ensemble version 2.0. Validation data is reported at the fixed majority-vote
boundary of zero; it no longer changes the decision threshold.

The algorithm matches the MATLAB training/testing logic, but models are not
expected to be numerically bit-identical because NumPy/SciPy and MATLAB differ
in random-permutation and linear-algebra implementations.
See `MATLAB_ENSEMBLE_COMPATIBILITY.md` for the function-by-function mapping and
the remaining reproducibility differences.

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
2026-09-27 10:15:42 | INFO     | Completed d_sub=400: OOB=0.1840, L=76, elapsed=8.2 s
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
