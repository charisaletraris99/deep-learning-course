# Gradient magnitude optimizer experiment

This branch contains the optimizer experiment code and its setup files. Assignment submissions, plant-health material and Word reports are kept on the `deep-learning-assigment-1` branch of this repository.

## Setup

Use Python 3.12. From this repository root in PowerShell:

```powershell
python -m venv DL_venv
.\DL_venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` is the original course environment snapshot shared with the assignment branch. The report builder additionally requires `python-docx`, included in `requirements.txt`.

## Run

```powershell
# Iris learning-rate search
.\DL_venv\Scripts\python.exe gradient-magnitude-experiment/run_experiment.py --record-test

# Iris comparison with the stored final-validation-accuracy rates
.\DL_venv\Scripts\python.exe gradient-magnitude-experiment/run_best_rates.py

# Digits search followed by its selected-rate comparison
.\DL_venv\Scripts\python.exe gradient-magnitude-experiment/run_digits.py

# One plot per optimizer with all tested rates
.\DL_venv\Scripts\python.exe gradient-magnitude-experiment/plot_test_accuracy_by_rate.py --results gradient-magnitude-experiment/results/YOUR_RUN_FOLDER

# Numerical checks
.\DL_venv\Scripts\python.exe -m unittest discover -s gradient-magnitude-experiment -p "test_*.py"
```

See [the experiment guide](gradient-magnitude-experiment/README.md) for equations, methodology and output descriptions. Iris and digits are bundled with scikit-learn. Generated results, checkpoints, images, reports and Python environments are not committed to this code branch. Paths to historical runs in the guide and JSON record describe the original local experiments; create new runs after cloning. `build_experiment_report.py` is an optional report builder for those original result folders and requires the referenced local histories and plots.
