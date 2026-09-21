# Deep Learning course

This folder follows the course setup guide, using Python 3.12 and TensorFlow/Keras on CPU.

## Run Python in VS Code

Open this folder in VS Code. If asked for a Python interpreter, select
`DL_venv\Scripts\python.exe` using **Python: Select Interpreter** from Ctrl+Shift+P.
Open `check_setup.py` and click **Run Python File** in the top-right corner.

## Terminal

Open a new terminal in VS Code (Terminal > New Terminal). If the environment is
not already active, run:

```powershell
.\DL_venv\Scripts\Activate.ps1
python check_setup.py
```

## Notebooks

Open `getting_started.ipynb`. Click **Select Kernel**, then choose the course
`DL_venv` environment (or **Python (Deep Learning)**). Click **Run All**.
This example uses generated data; it is a setup check, not an assignment solution.

## Packages

`requirements.txt` lists the requested packages. `requirements-lock.txt` records
the installed versions. Keep coursework here, but do not submit `DL_venv` with
your assignments.
