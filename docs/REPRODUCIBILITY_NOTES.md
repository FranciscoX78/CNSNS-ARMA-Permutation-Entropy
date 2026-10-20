# Scope and cautions

The goal is transparent regeneration of published numerical calculations,
not certification of all statistical modeling assumptions.

- **Synthetic A:** 270 independent Gaussian AR(2) realizations; two Gaussian references are not interchangeable. Original generator code and individual derived results are included.
- **Synthetic B:** presentation and derived CSVs are AR(1)-only, as decided by the authors. The AR(1)-only rerun preserves the generating-signal seed and parameters but has a different surrogate RNG consumption pattern from the recorded exploratory run. Frozen AR(1) results are definitive for the published figures.
- **IMS:** ready-to-use record-level CSVs include the fitted dominant-root modulus. Repeating all 170 windows per record × 984 records × 4 bearings from raw waveforms is slow. The original data must be obtained from NASA. The analysis uses local conditional sum-of-squares ARMA models with BIC-style order selection, not a formal full Gaussian ML fit.
- **Ordinal ties:** IMS excludes tied ordinal vectors within each window; the last two records are unsuitable and are not included in correlation figures.
- **Physiology:** ICP original signal not distributed; available upon reasonable request.
- **Inference:** record-level correlation does not prove Gaussianity, model adequacy, causality, or universal detection of nonlinearity or mechanical faults.
