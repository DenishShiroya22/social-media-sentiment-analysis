# XGBoost resource observations

The first full-fold pilot used 160 trees at the default histogram bin count and had not completed one fit after more than two minutes. It was interrupted before producing any TRAIN-CV score or accessing VALIDATION. The fixed search was then bounded to ten settings with 40–80 trees and max_bin=32: 30 serial TRAIN-CV fits plus one final fit, with two XGBoost threads per fit.

During a deeper full-fold fit on Windows, a process snapshot showed approximately 3.7 GB working set. A later snapshot between folds showed about 1.1 GB. These are observed snapshots, not a measured peak. The machine had approximately 16 GB physical RAM. Combined word and character TF-IDF stayed CSR sparse throughout; no dense feature conversion was used. The CSV search log and machine-readable metrics contain per-candidate fit times.
