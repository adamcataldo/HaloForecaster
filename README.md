Source code for the Halo paper

## Prerequisites

In order to run training and inference from this repo:

1. Results are written **outside this repository**, to a directory you choose. Run this to specify the directory:

   ```sh
   export HALO_RESULTS_DIR=~/some/results/dir
   ```
2. Install [uv](https://docs.astral.sh/uv/) on your path if it's not already there.
3. Install [git](https://git-scm.com) on your path if it's not already there.
4. Install third-party dependencies for code with no package on PyPI:

   ```sh
   uv run get_deps.py
   ```

## Running training and inference experiments

To sweep a model's hyperparameters on the
validation split, run:

```sh
uv run halo tune --model <model>
```

Each hyperparameter search space is defined in a configuration file like `src/halo/multi_head_timexer_sweep_configs.py`, for the swept dual-head TimeXer. The name to give for `<model>` is found from the `MODEL_NAME` specified in the file. In this case, use `MultiHeadTimeXer_sweep`. 

This command trains all configurations in the search spsace, records every trial, and publishes the lowest validation MSE of each dataset as that model's result. A trial already in the database at the same seed is skipped, so an interrupted sweep resumes where it stopped, and a dataset's winner appears only once every point of its grid has run.

To measure the model that performed best in validation on the test split, run:

```sh
uv run halo test --models <model...>
```
The same configuration file used for defining the hyperparameter search space also owns the winning configuration used for testing.

## Experimental setup

The experiment reproduces TimeXer's Table 2: day-ahead electricity price
forecasting on the five markets of the EPF benchmark, predicting 24 hours ahead
from a week of history.

| Market | Where | Exogenous series | Period |
|--------|-------|------------------|--------|
| `NP` | Nord Pool | grid load forecast, wind power forecast | 2013-01-01 – 2018-12-24 |
| `PJM` | PJM, USA | system load forecast, zonal COMED load forecast | 2013-01-01 – 2018-12-24 |
| `BE` | EPEX, Belgium | generation forecast, system load forecast | 2011-01-09 – 2016-12-31 |
| `FR` | EPEX, France | generation forecast, system load forecast | 2011-01-09 – 2016-12-31 |
| `DE` | EPEX, Germany | wind power forecast, zonal load forecast | 2012-01-09 – 2017-12-31 |

Each market is one hourly CSV of 52,416 rows — 2,184 days, six years to the day
— holding the day-ahead price and the two forecasts its operator publishes
alongside. The series are Lago et al.'s open-access benchmark [paper](https://doi.org/10.48550/arXiv.2008.08004).

**The forecasting task.** One example reads 168 hours of all three series and
predicts the next 24 hours of price alone — multivariate in, univariate out.
Windows advance an hour at a time, so every hour with a full week behind it and
a full day ahead of it is an example.

**The splits.** Each series is cut chronologically: the first 70% trains, the
next 10% validates, the last 20% tests — roughly 4.2 years, 7 months and 15
months. A split's windows start one look-back early, so the first validation
target is the hour after the training boundary and its input week is the tail of
the training series. No hour goes unpredicted, and no target hour belongs to two
splits.

**Standardization.** Every column is standardized to zero mean and unit variance
with statistics fit on the training rows alone, then applied to all three
splits. MSE and MAE are reported in those standardized units, pooled over every
predicted hour. That is the convention TimeXer's Table 2 uses, so the numbers
here are comparable to the published ones — they are not euros per megawatt-hour.

**Validation and test.** Training runs up to 50 epochs and stops early once
validation MSE has gone five epochs without improving. Validation MSE at the
best epoch is what a sweep chooses on and what the validation results record.
`test` scores the other split: it retrains each model from the same
configuration at the same derived seed, stops on validation exactly as before,
and scores the weights from the epoch validation chose against the test split
once. Every test run prints its recomputed validation numbers beside its test
numbers, which is the evidence that the retraining reproduced the published run.

**The `Avg` row.** Table 2 closes with an average across its five markets, so
the results reproduce it — an unweighted mean of the five, whose cells fill in
only once all five are publishable at a single shared seed. The row is always
there and empty until then. It occupies the market column without being a
market, which is worth knowing if you read that column as a list of markets.

## Results

Each split owns its own files, so the two sets never mix and neither can
overwrite the other. The results directory ends up holding validation results and test results:

```
Core validation results:
    results.db

Viewer-friendly validation results:
    short_term_exogenous_results.csv
    short_term_exogenous_results.xlsx

Core test Results:
    test_results.db
    
Viewer-friendly test results:
    short_term_exogenous_test_results.csv
    short_term_exogenous_test_results.xlsx
```

The `.db` files are source-of-truth SQLite tables for results. The `.csv` and `.xlsx` files are rollups of the results, and are meant to be viewer-friendly results for a human to look at.

## Limitations

**CUDA is not supported.** Training uses Apple Silicon's MPS backend when it is
available and the CPU otherwise; nothing looks for a CUDA device. On an NVIDIA
machine every run is still correct, but it trains on the CPU and says nothing
about the idle GPU. Two models stay on the CPU wherever they run: GCGNet and the
dual-head GCGNet pin it.

**Parallelism numbers were measured on one machine.** Each configuration carries
a number saying how many copies of itself can train at once, and the scheduler
divides the machine according to it rather than by a global concurrency limit. The numbers checked in here were measured on Apple
Silicon, so on other hardware they may be wrong.

If runs thrash — instances dying, or total throughput falling as more of them
start — measure the numbers again. For each dataset (NP, PJM, BE, FR, DE) and
the model configuration you care about (TimeXer, say):

1. Measure it:

   ```sh
   uv run halo calibrate --model TimeXer --dataset NP --horizon 24
   ```

2. Take the number from the last line of the output:

   ```
     N    per-instance steps/s  aggregate  verdict
     1                    3.41       1.00  worth it
     2              3.05, 3.02       1.78  worth it
     4  0.71, 0.68, 0.70, 0.69       0.82  slower than sequential

   suggested parallelism: 2
   ```

3. Put it into `EXOGENOUS_PARALLELISM` in the model's module:

   | Model | Module |
   |-------|--------|
   | `TimeXer`, `MultiHeadTimeXer` | `src/halo/timexer_configs.py` |
   | `MultiHeadTimeXer_sweep` | `src/halo/multi_head_timexer_sweep_configs.py` |
   | `TimeXer_sbs` | `src/halo/timexer_sbs_configs.py` |
   | `CrossLinear` | `src/halo/crosslinear_configs.py` |
   | `CrossLinear_sbs` | `src/halo/crosslinear_sbs_configs.py` |
   | `MultiHeadLinear` | `src/halo/multi_head_linear_configs.py` |
   | `GCGNet` | `src/halo/gcgnet_configs.py` |
   | `MultiHeadGCGNet` | `src/halo/multi_head_gcgnet_configs.py` |

