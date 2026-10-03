# MATLAB ensemble compatibility

The classifier is based on the official DDE Lab Ensemble Classifier version
2.0 (`ensemble_training.m` and `ensemble_testing.m`, September 2013), which
implements the method from:

J. Kodovský, J. Fridrich, and V. Holub, “Ensemble Classifiers for
Steganalysis of Digital Media,” *IEEE Transactions on Information Forensics
and Security*, 7(2), 432–444, 2012.

## Implemented equivalences

| MATLAB procedure | Python implementation |
| --- | --- |
| Separate synchronized `Xc` and `Xs` matrices | Pair rows by matching paths below `cover/` and `stego/` |
| One bootstrap vector shared by `Xc` and `Xs` | Paired cover/stego bagging, always enabled |
| Random `d_sub` features without replacement | One random feature permutation per base learner |
| FLD from `sigC + sigS + 1e-10 I` | Same within-class covariance sum and diagonal regularization |
| Remove equal constant features | Set their FLD weights to zero |
| Select each FLD threshold by minimum `FA + MD` | Same sorted-projection threshold search, including orientation reversal |
| Accumulate OOB sign decisions | Same per-pair OOB majority-vote estimate |
| Automatic stopping for `L` | Same moving-average epsilon-tube rule |
| Automatic search for `d_sub` | Port of the staged OOB grid search in `update_search` |
| Testing by sum of `sign(Xw-b)` | Same binary majority vote and random resolution of exact ties |

The training JSON records the OOB trajectory, all evaluated subspace sizes,
the selected OOB error, `d_sub`, and `L`.

## Deliberate implementation differences

- NumPy/SciPy replace MATLAB matrix operations. Floating-point results can
  differ slightly, particularly for nearly singular covariance matrices.
- Both implementations use MT19937-style pseudorandom streams, but their
  permutation routines are not guaranteed to produce identical sequences.
- MATLAB testing uses the current global random stream for tied votes. This
  implementation uses `--tie-break-seed` (default `1`) so repeated evaluations
  and saved metric files are reproducible.
- Features and trained models use compressed NumPy and Python pickle formats
  instead of MATLAB `.mat` structures.

These differences prevent byte-for-byte model equality but do not change the
classifier architecture or its training and majority-voting rules.
