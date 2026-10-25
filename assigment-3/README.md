# Assignment 3 Exercise 1 Question 4

`imdb_lstm.py` now runs the selected **three-layer GRU**, retaining the original filename. The nine-model comparison selected it using validation results only: at epoch 3 it achieved both the highest validation accuracy (89.38%) and the lowest validation loss (0.2724).

## Comparison evidence

The completed comparison is in `results_recurrent_comparison/20261015_150230_305523`. Each candidate trained for 15 epochs. The values below pair the highest validation accuracy with the loss at the same epoch.

| Model family | Layers at highest accuracy | Epoch | Validation accuracy | Related loss |
| --- | ---: | ---: | ---: | ---: |
| SimpleRNN | 1 | 4 | 87.72% | 0.3522 |
| GRU | 3 | 3 | 89.38% | 0.2724 |
| LSTM | 2 | 2 | 88.90% | 0.2727 |

The GRU's advantage over the best LSTM is small: 0.48 percentage points and approximately 0.0003 loss. This is evidence from one split and seed, not proof of a universally better architecture. See `RECURRENT_COMPARISON.md` for the nine-model experiment.

## Run the selected model

From the repository folder:

```powershell
.\DL_venv\Scripts\python.exe .\assigment-3\imdb_lstm.py
```

From `assigment-3`:

```powershell
..\DL_venv\Scripts\python.exe .\imdb_lstm.py
```

The full run trains from scratch for **35 epochs**, without early stopping, so later overfitting remains visible. It uses 35,000 training, 5,000 validation and 10,000 test reviews, with a stratified split and seed 42. All models use the same split. Test results are recorded for the requested graphs, never used for checkpoint selection. Earlier project work already inspected this test split.

## Architecture and settings

Word IDs enter an embedding with 10,000 entries and 32 dimensions. Three GRU layers each have 32 units. The first two return the full sequence and the third returns one vector. One sigmoid output predicts positive-review probability; 0.5 is the classification threshold.

Reviews retain the last 500 tokens and use masked right padding. Training uses Adam with learning rate 0.001, gradient clipping norm 1.0, batch size 128 and binary cross-entropy. These are practical shared comparison settings, not claimed optima. The model has 339,041 parameters.

GRU gates can help retain useful context and reduce vanishing-gradient difficulties. Additional layers can combine sequence features, but also increase complexity and overfitting risk. The comparison, rather than this intuition alone, supports choosing three GRU layers here.

## Saved outputs

New outputs go to `results_gru_3_layers`. Earlier LSTM results remain in `results_lstm`. Running the selected script again replaces its output files.

- `config.json`: settings and the comparison run used for selection.
- `history.json`: end-of-epoch training, validation and test metrics.
- `accuracy.png` and `loss.png`: all 35 epochs, marking the minimum-validation-loss epoch.
- `model.keras`: checkpoint with the lowest validation loss, updated during training.
- `final.keras`: epoch-35 model, saved separately when training finishes.
- `selected_metrics.json`: selected epoch and its train/validation/test scores, written when training finishes.
- `classification_report.txt` and `confusion_matrix.png`: test predictions of the selected checkpoint, not necessarily epoch 35.

Accuracy and loss graphs describe the whole training trajectory. The final classification report evaluates the restored best-validation-loss model. This distinction matters when later epochs overfit.

The Word report includes the comparison at the beginning and retains the original LSTM experiment as the initial baseline. The completed 15-epoch comparison results must not be mistaken for the separate 35-epoch GRU results.
