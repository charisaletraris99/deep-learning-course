# Mean absolute gradient experiment

This project tests Charis Aletraris's proposed optimizer using TensorFlow. The comparison methods are optimizers, all training the same neural network.

## Exact rules

For each minibatch, compute the gradient of mean cross-entropy. Define
`m_t = sum_i(abs(g_t,i)) / N`, where N counts **all trainable scalar parameters**, including biases. This is a global mean, not a mean of layer means or per-example absolute gradients.

- `mag` (your original idea): `eta_t = base_lr * m_t`.
- `mag_floor` (your added variant): `eta_t = base_lr * max(m_t, 1)`. It uses the original multiplier above 1, and a multiplier of 1 at or below 1. The logged `mean_abs_gradient` remains the raw mean; `scalar_lr` records the resulting learning rate. Its default rate grid matches SGD because it behaves like SGD whenever `m_t <= 1`.
- `inverse_mag` (professor's suggested variant): `eta_t = base_lr / (m_t + 1e-8)`.
- `sgd`: fixed learning rate, no momentum.
- `adam`: standard TensorFlow Adam defaults apart from learning rate.
- `adagrad_norm`: `b_t^2 = b_(t-1)^2 + sum_i(g_t,i^2)`, starting at `b_0=1`; `eta_t = base_lr / b_t`, using the newly accumulated value.

The scalar rules update `theta <- theta - eta_t * g_t`. No clipping, momentum, weight decay, smoothing or learning-rate cap is added to your rule. Optional `rmsprop` and coordinate-wise `adagrad` baselines are also available through `--methods`.

AdaGrad-Norm follows the recurrence in Ward, Wu & Bottou (2019), [AdaGrad Stepsizes: Sharp Convergence Over Nonconvex Landscapes](https://proceedings.mlr.press/v97/ward19a.html). This implementation is a small independent experiment, not a reproduction of that paper. LARS and Polyak methods were related-literature suggestions in the email; they are not implemented here.

## Run from the course folder (PowerShell)

Use the existing course environment; no dataset download is needed.

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\run_experiment.py
```

Default: 40 epochs, 20 seeds (42-61), batch size 32, six optimizers and three learning-rate candidates per optimizer (360 training runs). A quick functional check:

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\run_experiment.py --epochs 2 --seeds 1
.\DL_venv\Scripts\python.exe -m unittest discover -s .\gradient-magnitude-experiment -p "test_*.py"
```

Assignment-length comparison with all available methods:

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\run_experiment.py --methods mag mag_floor inverse_mag sgd adam adagrad_norm rmsprop adagrad
```

## Experimental controls

The dataset and defaults match `exercise1a.py` in `Assignment1_Charis_Aletraris.zip`, which generated the assignment report referenced by `report_render`. That folder contains report artifacts rather than a standalone dataset file; Iris is loaded locally using scikit-learn.

The 150 Iris flowers are split into **108 training (72%), 12 validation (8%), and 30 test (20%)** samples. The original non-stratified `train_test_split(test_size=0.2, random_state=42)` supplies 120 non-test rows; its last 12 rows form validation, matching Keras `validation_split=0.1` before shuffling. Exact row indices are saved. The network is **4 inputs -> 16 tanh -> 3 softmax**, with one-hot labels and mean categorical cross-entropy. Defaults are **40 epochs, batch size 32, and 20 runs with seeds 42-61**.

To reproduce the original preprocessing, StandardScaler is fitted on the 120 non-test samples before reserving validation. Thus validation features contribute to scaling statistics, as in the assignment; the test set does not. A stricter future experiment would fit on only 108 training rows, but would no longer exactly match this preprocessing. Scaler statistics are saved in `scaler.npz`.

For each seed, every method and candidate starts with the same weights and receives the same minibatch order. Each method gets three rate candidates by default. The best validation-loss epoch and rate are selected separately per seed and method; only that checkpoint is evaluated on the test set. Thus summary results describe the validation-tuned procedure, not one universally chosen rate. Test data is never used for rate or epoch selection. Unlike the assignment's final-epoch tables and per-epoch test plots, this optimizer experiment retains validation-selected checkpoints and evaluates the test set only for the selected candidate. The epoch budget and data partitions match; this reporting policy is unchanged. The custom training loop also uses its own reproducible batch shuffle, so results need not be numerically identical to Keras model.fit. Seeds vary initialization and batch order, not the data split. Standard deviation across seeds does not measure uncertainty across datasets.

Grid ranges are exploratory, not guaranteed optimal. Rates differ because their units and effects differ. Expand a range if validation prefers its boundary. For a custom grid, pass `--grid path/to/grid.json`, with content such as:

```json
{"mag": [0.1, 1, 10, 100], "inverse_mag": [0.000001, 0.00001, 0.0001, 0.001]}
```

Keep candidate counts comparable for a fair search budget. Freeze search decisions before using test results for conclusions. Non-finite runs are marked `diverged` and excluded from selection, even if an earlier epoch was finite. Failures remain in `runs.csv`; methods with no successful seed are absent from summary.csv, so inspect failures too. A finite but poor run is still recorded as completed.

## Outputs

Each run creates a new `results/<timestamp>/` directory containing:

- `config.json`, `scaler.npz` and `split_indices.npz`: settings, versions and exact split.
- `runs.csv`: every candidate's validation result, runtime and failure status.
- `history.csv`: epoch metrics for every candidate, including batch-averaged gradient magnitude, scalar learning rate and actual parameter update L2 norm.
- `selected.csv`: chosen rate/epoch and test results for each seed and method.
- `summary.csv`: mean test accuracy, sample standard deviation and successful-seed counts. Standard deviation is undefined with only one seed.
- `comparison.png`: validation-selected curves averaged across successful seeds.
- `*.weights.h5`: selected checkpoint weights, loadable into `model_for(seed)`.

For Adam/RMSProp/AdaGrad, `scalar_lr` is the configured base rate; those methods use coordinate-specific scaling, so it is not their effective per-coordinate rate. Actual update norm is logged for all methods. Runtime includes graph tracing and evaluation, so it is not a controlled optimizer throughput benchmark. Full curves continue beyond the selected best epoch.

## What the experiment can tell you

For a one-dimensional loss `L(theta)=theta^2/2`, your rule gives `theta_next = theta - base_lr*abs(theta)*theta`. Its step magnitude is proportional to the square of the gradient. The denominator variant's step magnitude is approximately constant when the gradient is appreciably above epsilon, so it can oscillate near a minimum. Neither is guaranteed to win.

This is a starting experiment on one small dataset and architecture. It cannot establish novelty, general superiority or a convergence theorem. Follow up with more tasks, architectures, training seeds and fixed search budgets before making a research claim. Changing the loss scale or parameterization can materially change these comparisons.

## Fixed best-rate comparison: three accuracy plots

Run the new runner from the course folder:

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\run_best_rates.py
```

It uses one fixed base learning rate per optimizer from the `BEST_BASE_RATES` dictionary near the top of `run_best_rates.py` (`best_learning_rates.json` is a record of the selection and is no longer read by this runner), selected from the completed Iris experiment `results/20261006-093705-631888`. Selection maximizes mean validation accuracy at epoch 40 across all 20 seeds (ties prefer the smaller rate). This replaces the earlier best-validation-loss criterion; test results are not used for selection. The selected rates are sgd=1, adam=0.01, mag=10, mag_floor=1, inverse_mag=0.01, adagrad_norm=1. These are the best among the tested candidates, not guaranteed global optima. All six optimizers present in that experiment are included.

The default is 40 epochs, batch size 32 and 20 seeds (42-61), with the same Iris split and model: 120 training runs total, without another rate search. It records training, validation and test accuracy on each complete partition after every epoch and averages over the 20 runs. No checkpoint replacement or early stopping changes these curves. Test evaluations are for reporting only and do not affect parameter updates or selection. Reusing the prior seeds reproduces that experimental setup rather than providing an independent confirmation study.

Each invocation creates a fresh `results_best_rates/<timestamp>/` folder containing exactly three PNG plots, each with all six optimizer curves:

- `training_accuracy.png`
- `validation_accuracy.png`
- `test_accuracy.png`

It also saves individual histories, epoch means, final-epoch accuracies, run statuses and reproducibility settings. Plots are saved, not automatically opened. A non-finite run stops with saved diagnostics instead of silently dropping seeds from the average. SGD and MAG-floor may overlap when the raw mean absolute gradient stays below 1.

For a short functional check use `--epochs 2 --seeds 42 43`. This does not change the stored rates or the defaults.

## Handwritten digits benchmark

Run the complete new workflow from the course folder:

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\run_digits.py
```

The scikit-learn handwritten digits dataset contains 1,797 images, each 8x8 pixels, across 10 digit classes. It is bundled with the installed scikit-learn package, so no download or extra installation is required. Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

The fixed stratified split contains 1,293 training, 144 validation and 360 test samples (approximately 72/8/20 percent, following the assignment proportions). Pixels are divided by the known maximum 16; no normalization statistics are fitted to held-out data. The network is 64 inputs -> 16 tanh -> 10 softmax, with 1,210 trainable parameters. All six optimizers use the same network and paired initialization and batch order within each seed. This is a new benchmark, not a reproduction of the earlier preliminary digits runs that used a different architecture and split.

Defaults remain 40 epochs, batch size 32 and seeds 42-61. The workflow first searches the four current GRIDS values per optimizer (480 training runs), chooses the highest mean final validation accuracy for each optimizer, and then runs the fixed-rate comparison (120 runs) with three per-epoch accuracy plots. Iris rates are not reused. Grids are starting candidates; boundary winners can motivate a separate wider search. Runs with non-finite values are recorded; a candidate must complete every seed to be eligible. If any method has no eligible candidate, the workflow stops rather than silently excluding it.

Search results go to `results_digits/<timestamp>/`, including `best_final_rates.json`. The three final plots and accuracy tables go to `results_digits_best_rates/<timestamp>/`. Iris results and defaults remain available through the original commands. The full workflow may take longer than Iris because each epoch has more minibatches.

A quick functional check:

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\run_digits.py --epochs 2 --seeds 42
```

To run just the search, use `run_experiment.py --dataset digits`. To rerun a completed search's selected rates, use `run_best_rates.py --selection <path-to-best_final_rates.json>`. When a selection file is supplied, its dataset, epochs, batch size and seeds control the rerun; the in-code Iris rate dictionary is not used. The original search's `summary.csv` still describes per-seed best-loss checkpoints; `best_final_rates.json` and the subsequent three plots use final mean validation accuracy.

## One test accuracy plot per optimizer with all base rates

Run:

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\plot_test_accuracy_by_rate.py
```

By default this chooses the newest completed compatible Iris or digits search. To choose a specific experiment:

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\plot_test_accuracy_by_rate.py --results .\gradient-magnitude-experiment\results\20261006-093705-631888
```

Use a `run_experiment` search folder with `config.json`, `history.csv` and all the tested grids, not a selected-rate folder. The script creates one PNG per optimizer (for example `mag_test_accuracy.png`), with one curve per base rate. Every point is the arithmetic mean test accuracy across the source experiment's seeds at that epoch. It also saves `mean_test_accuracy_by_epoch.csv` with mean, standard deviation and seed count. Outputs are in a fresh `results_rate_plots/<timestamp>/` folder.

Older search histories did not store test accuracy at every epoch. A best-checkpoint test score cannot reconstruct that trajectory. For those histories this script repeats all saved rates, seeds, epochs and batch settings with the matching dataset/model and verifies the split indices, recording the missing measurements in the new output folder. These are explicitly marked as recomputed measurements using current code and TensorFlow, rather than recovered historical values. Existing results are not overwritten. If measurements already exist, it plots them directly without training. Incomplete histories or failed runs do not silently become averages over fewer seeds; plotting fails with saved diagnostics.

For future searches, add `--record-test` to `run_experiment.py` to store these measurements while training, avoiding the repeat:

```powershell
.\DL_venv\Scripts\python.exe .\gradient-magnitude-experiment\run_experiment.py --dataset digits --record-test
```

Test curves are descriptive; keep base-rate selection based on validation data. The script does not select a winner based on test curves.

## Letter Recognition MNIST and Fashion MNIST

Choose near the top of `run_experiment.py`:

```python
DATASET = "letter"  # or "mnist", "fashion_mnist", "iris", "digits"
```

All three new datasets are downloaded on first use and cached under `gradient-magnitude-experiment/datasets/`. Subsequent runs use the cached files. To download and validate them without training, run `experiment_datasets.py`.

| Name | Samples | Features | Classes | Train / validation / test |
|---|---:|---:|---:|---|
| letter | 20,000 | 16 | 26 | 14,400 / 1,600 / 4,000 |
| mnist | 70,000 | 784 | 10 | 54,000 / 6,000 / 10,000 |
| fashion_mnist | 70,000 | 784 | 10 | 54,000 / 6,000 / 10,000 |

Letter uses a fixed stratified 80/20 split, then reserves 10% of the non-test rows for validation. It divides known-range features by 15. MNIST and Fashion-MNIST preserve their official test sets and take a stratified 6,000-sample validation set from the official training portion. Images are flattened from 28x28 to 784 features and divided by 255. No preprocessing statistics are learned from validation or test data for these datasets. Seed 42 controls splitting.

The model continues to use 16 tanh hidden units, with automatic input and output dimensions. This keeps the existing optimizer experiment simple; it is not intended as a state-of-the-art image classifier. Training defaults remain 40 epochs, batch size 32 and 20 seeds. These larger searches will take considerably longer than Iris. Tune rates for each dataset rather than reusing Iris's chosen rates.

```powershell
.\DL_venv\Scripts\python.exe gradient-magnitude-experiment/run_experiment.py --dataset letter --record-test
```

New runs save to `results_letter/letter_<timestamp>`, `results_mnist/mnist_<timestamp>` or `results_fashion_mnist/fashion_mnist_<timestamp>`. Each completed search exports `best_final_rates.json` using final mean validation accuracy if every method has an eligible candidate. Run the fixed-rate comparison with:

```powershell
.\DL_venv\Scripts\python.exe gradient-magnitude-experiment/run_best_rates.py --selection "PATH_TO_SEARCH/best_final_rates.json"
```

The plotting script accepts those search directories through `--results`. Use `--record-test` on the search to avoid retraining just to obtain per-epoch test curves. Test metrics are for reporting, not rate selection.

Sources:
- Letter Recognition: https://archive.ics.uci.edu/dataset/59/letter+recognition (UCI ZIP)
- MNIST: https://www.tensorflow.org/datasets/catalog/mnist (TensorFlow/Keras hosted NPZ, verified against the Keras SHA-256)
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist (TensorFlow/Keras hosted IDX gzip files)

Letter's original feature records are also exported as `datasets/letter/letter-recognition.csv`. Image datasets remain in their original compressed formats. Each local dataset folder contains a `metadata.json` with dimensions and split counts. Dataset files are ignored by Git.
