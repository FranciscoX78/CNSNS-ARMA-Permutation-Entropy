# Linear-Gaussian permutation-entropy references for empirical time series

**Companion research code and derived results** for the manuscript
*Permutation Entropy in Linear Gaussian ARMA Processes: A Reciprocal-Root Reference for Empirical Signals* (CNSNS-D-26-03130, revised manuscript).

This repository documents the numerical calculations and provides the processed
results used in two controlled synthetic experiments and in the NASA IMS Set 2
bearing-vibration analysis. **It does not contain intracranial-pressure (ICP) recordings.**
The original ICP recordings are available from the corresponding author upon
reasonable request (subject to applicable institutional conditions).

## Quick start — reproduce tables and plots without downloading NASA data

Requirements: Python 3.11+ (Python 3.11/3.12 suggested), NumPy, pandas,
SciPy, Matplotlib, and Numba. From this repository's root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_results.py
python scripts/plot_paper_figures.py --part all
```

The verification script checks key numerical summaries (including the main
AR(2) synthetic correlation and the four NASA/IMS record-level correlations).
New plots appear under `figures/regenerated/`. They reproduce the **underlying
series and statistics** but are not promised to be pixel-identical to the
paper's approved figure artwork in `figures/`.

## Contents

| Path | Description |
|------|-------------|
| `scripts/generate_experiment_A.py` | Full original Gaussian AR(2) simulation and fitted-reference analysis, adapted to relative output paths. |
| `scripts/generate_experiment_B_AR1.py` | AR(1)-only re-run of the nonlinear experiment. See the random-stream caveat below. |
| `scripts/plot_paper_figures.py` | Generate the accepted four-panel synthetic figures and IMS main/appendix figures from committed CSVs. |
| `scripts/prepare_ims_set2.py` | Convert the original NASA IMS Set 2 text files to arrays and a manifest expected by the fitting code. |
| `scripts/fit_ims_windows.py` | Original low-order per-window ARMA fitting and Gaussian entropy calculations for m=4,5,6, adapted to relative paths. |
| `scripts/verify_results.py` | Numerical consistency checks against the published derived CSVs. |
| `scripts/summarize_ims_poles.py` | Recover record-level dominant reciprocal roots from freshly fitted m=4 windows. |
| `data/experiment_A/` | 270 independent AR(2) realizations summarized by empirical PE, fitted population Gaussian Hlin and matched-length surrogate PE. |
| `data/experiment_B/` | AR(1) results only; nine nonlinear perturbation levels with 30 realizations each. |
| `data/ims/record_level_m4/` | The 982 usable record-level summaries used in the four-bearing and single-bearing figures, including dominant-pole modulus. |
| `data/ims/summary_m456/` | Record summaries for m=4,5,6, including 984 original records and quality flags. |
| `figures/` | Approved paper figure PNGs. |

The submitted manuscript also contains theoretical AR/MA/ARMA entropy
landscapes and the original ICP application; those are not being claimed as
reproducible from the materials here.

## Synthetic Experiment A — controlled Gaussian AR(2)

Nine independent regimes, signed dominant reciprocal root
`r in [0.95, 0.80, 0.60, 0.35, 0, -0.35, -0.60, -0.80, -0.95]`,
second root `0.35*r`. The generating process is Gaussian AR(2):
`phi1=r+r2`, `phi2=-(r*r2)`. Each regime has 30 independently
generated stationary signals, `N=3600` observations after a 1000-point
burn-in, `m=6`, `tau=1`. The figure concatenates one representative
realization from each regime **only for visualization**.

```bash
python scripts/generate_experiment_A.py
```

This recreates outputs in `data/regenerated_A/`. Its full numerical
Gaussian entropy construction uses Sobol quasi-Monte Carlo sampling and
25 matched-length Gaussian surrogates per fitted window; full regeneration
can take a substantial amount of CPU time.

**Two distinct references must not be conflated:**
`Hlin_fit_gaussian` estimates the Gaussian population entropy of the fitted
AR(2); `Hlin_fit_matched_N` is the mean finite-sample PE estimator applied
to fitted-model surrogate signals of equal length. The error shown in the
approved Figure A is against **population Hlin**, whereas the smaller
finite-sample-matched error is discussed in the manuscript.

## Synthetic Experiment B — nonlinear perturbation, AR(1) reference

`x_t = 0.55*x_(t-1) + alpha*sin(x_(t-1)) + epsilon_t`, with
`epsilon_t ~ N(0,1)`, and 9 levels of alpha. There are 30 realizations per
level, using `m=6`, `tau=1`, and `N=3600`. The *published numerical
comparison* uses **only fitted Gaussian AR(1)** as the reference, in line
with the final manuscript. The paper figure compares sample PE with
population Hlin in panel B, and with finite-length matched AR(1)
surrogates in panels C–D.

```bash
python scripts/generate_experiment_B_AR1.py
```

**Reproducibility precision note:** The frozen per-window and surrogate
CSVs reproduce the numerical results in the revised paper. The standalone
AR(1)-only generator uses the original signal and QMC seeds but does not
replay auxiliary random-number draws from exploratory calculations
excluded from the final paper. Therefore independently rerunning this
script may produce slightly different matched-surrogate sample means and
quantiles; this is Monte Carlo variation, not a change in the model.

## NASA IMS Set 2 — publicly available empirical data

The four simultaneous vibration channels were provided by the Center
for Intelligent Maintenance Systems, University of Cincinnati, and
are publicly distributed by NASA. The raw archive is **not included**.

- Official landing page: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- NASA's `4. Bearings` archive: https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
- NASA Open Data Portal: https://data.nasa.gov/dataset/ims-bearings

**Source citation:** Lee, J., Qiu, H., Yu, G., Lin, J., & Rexnord Technical
Services (2007). *Bearing Data Set*, IMS, University of Cincinnati, NASA
Prognostics Data Repository, NASA Ames Research Center.

To reproduce calculations from the original waveforms:

1. Download the NASA bearing archive and extract **Set 2** until you have
   the 984 timestamp-named text files (e.g. `2004.02.12.10.32.39`).
2. Convert these into the exact matrix layout used during analysis:

   ```bash
   python scripts/prepare_ims_set2.py --raw-dir data/raw/2nd_test --out-dir data/intermediate
   ```

   This step checks all 984 snapshots and their `20,480 x 4` numeric shape.
   If files are nested further than this example, point `--raw-dir` to a
   parent directory: the converter searches recursively.
3. Run the original low-order-model calculation:

   ```bash
   # Linux/macOS
   IMS_WORK_DIR=data/intermediate python scripts/fit_ims_windows.py m=4 b=1,2,3,4
   # Windows PowerShell
   $env:IMS_WORK_DIR='data/intermediate'; python scripts/fit_ims_windows.py m=4 b=1,2,3,4
   ```

4. Derive the record-level pole summaries used for the paper figures:

   ```bash
   python scripts/summarize_ims_poles.py --input-dir data/intermediate --output-dir data/regenerated_ims_record_tables
   ```

   For the sensitivity analysis, repeat the fit with `m=5` and `m=6`.

**Important:** `fit_ims_windows.py` is computationally expensive and has
not been rerun end-to-end here on the NASA raw data, because those
waveforms are not included in this archive. The code is the archived
estimation method with file paths made portable. The ready-to-use
record summaries and their checked numerical correlations are included.

The empirical main figure uses Bearing 1 with `m=4`, `tau=1`,
`N=5*m!=120`, and 982 usable recordings. Four-bearing results are in
the appendix. For `m=5` and `m=6`, the study reports numerical and
model-diagnostic limitations in the text rather than adding figures.

The mean dominant reciprocal-root modulus is defined **only** for
selected models with an autoregressive component; MA-only windows
contribute no artificial pole. Record-level symmetric error is
`200*abs(PE-Hlin)/(abs(PE)+abs(Hlin))` (percent).

**Model adequacy:** Many local ARMA fits do not satisfy residual
whiteness diagnostics, especially at higher embedding dimensions. High
correlations between empirical PE and Hlin are *not* independent
validation of the fitted Gaussian ARMA assumption.

## Data availability and reuse

- NASA raw vibration data: access from the **official NASA source**.
- Synthetic results, processed NASA summaries and code: contained here.
- ICP original data: available from the corresponding author upon
  reasonable request, **not publicly redistributed**.
- No claim is made that the GitHub repository supplies raw ICP data or
  reconstructs that analysis from publicly downloaded recordings.
- Repository code authors retain copyright unless an appropriate
  software licence is selected and added; third-party data may have
  independent terms. See `docs/PUBLISHING_CHECKLIST.md`.

## References

- Bandt, C., and Pompe, B. (2002), Permutation entropy: a natural complexity measure for time series, *Physical Review Letters*.
- Lee et al. (2007), NASA/IMS Bearing Data Set (see official NASA link above).
- Redelico, F. O., Traversaro, F., Oyarzabal, N., Vilaboa, I., & Rosso, O. A. (2017). Evaluation of the status of rotary machines by time causal information theory quantifiers. *Physica A*, *470*, 321–329. https://doi.org/10.1016/j.physa.2016.05.031

## Status

Prepared for author review; **not yet uploaded to GitHub**, not assigned a release tag or DOI, and not yet cited from the final manuscript. See the publishing checklist.
