# Assignment 2

## Exercise 1

The original code is in `excercise 1/`. The data-loading code is unchanged.
The existing Word report and PDF brief remain in this directory. The existing
Exercise 1 graphs remain in `outputs/exercise_1_three_conv_gelu/`.

## Exercise 2

The residual CNN is in `excercise 2/`. Read its [explanation](excercise%202/README.md)
before running the experiment. Its full training run and comparison are pending.

From the repository root in PowerShell:

```powershell
.\DL_venv\Scripts\python.exe "assignment-2/excercise 1/train_cnn.py"
.\DL_venv\Scripts\python.exe "assignment-2/excercise 2/train_residual.py"
```

Both scripts use 40 epochs, batch size 128, Adam with learning rate 0.001, and
seed 42. Both write only `learning_curves.png` and `confusion_matrix.png` to
separate output folders. Running a script again replaces its two graphs.
The report can use the final metrics printed in the terminal.
