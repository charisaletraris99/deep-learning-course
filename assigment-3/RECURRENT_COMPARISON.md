# Comparing recurrent models on IMDB

`compare_rnn_models.py` trains nine independent models from scratch:

| Model type | Experiments |
| --- | --- |
| SimpleRNN | 1, 2 and 3 recurrent layers |
| GRU | 1, 2 and 3 recurrent layers |
| LSTM | 1, 2 and 3 recurrent layers |

The layer count refers only to the recurrent layers. Every model also has one embedding layer and one sigmoid output layer. When stacking recurrent layers, each intermediate layer returns the whole sequence so the next layer can process it; the last recurrent layer returns one vector for classification.

## Run

From the `assigment-3` folder:

```powershell
..\DL_venv\Scripts\python.exe -u .\compare_rnn_models.py
```

From the repository folder:

```powershell
.\DL_venv\Scripts\python.exe -u .\assigment-3\compare_rnn_models.py
```

The script runs all nine experiments sequentially, for 15 epochs each. This is 135 training epochs, and deeper models take longer per epoch. On a CPU this can take several hours. There is no early stopping: the later epochs show how overfitting develops. Every launch creates a new timestamped run folder. It starts new training rather than resuming an interrupted run.

## Shared settings

- Exactly the same stratified split for all experiments: 35,000 training, 5,000 validation and 10,000 reserved test reviews. The split uses seed 42 and preserves the positive/negative balance.
- Training reviews update weights. Validation reviews only measure performance and determine which checkpoint is saved as best. Test reviews are not evaluated in this comparison.
- Vocabulary of 10,000 IDs; last 500 tokens per review; right padding with zeros and masking.
- Embedding dimension 32, and 32 units in every recurrent layer.
- Adam, learning rate 0.001, gradient clipping norm 1.0, batch size 128, binary cross-entropy and a sigmoid output with threshold 0.5 for accuracy.
- Each model starts with seed 42. This gives a reproducible comparison setup, but one seed is not enough to establish a robust ranking.
- No additional dropout or weight regularization, matching the original baseline. Both training and validation curves are evaluated at the end of each epoch using the same current weights.

Only architecture and depth are deliberately varied. Equal unit counts do not imply equal parameter counts: SimpleRNN, GRU and LSTM have different internal structures, and adding layers increases capacity. Parameter counts are recorded in the summary, so this is a comparison at fixed width and training settings, not a comparison at equal parameter or compute budgets.

## Figures and saved results

Inside `results_recurrent_comparison/<run timestamp>/`:

- `simplernn_comparison.png`: one figure comparing 1, 2 and 3 SimpleRNN layers.
- `gru_comparison.png`: one figure comparing 1, 2 and 3 GRU layers.
- `lstm_comparison.png`: one figure comparing 1, 2 and 3 LSTM layers.

Each figure has accuracy on the left and binary cross-entropy on the right. Blue, orange and green represent 1, 2 and 3 layers. Dashed lines are training; solid lines are validation. Figures are refreshed after each epoch, so unfinished experiments have shorter curves. The title states how many of the three experiments are complete. A model family's figure appears once its first epoch finishes.

`histories.json` contains all measured curves and completion flags. `summary.csv` records parameter counts, completed epochs, the lowest-validation-loss epoch, its validation accuracy, and final accuracy. `config.json` records the shared settings. `split_indices.npz` records the exact dataset membership.

Each experiment's subfolder contains `best.keras` (lowest validation loss so far) and, once it finishes, `final.keras` (epoch 15). The selected epoch is based on validation loss, not maximum test accuracy. The final console output identifies the lowest validation loss across the nine experiments.

## Reading the comparison

Look for lower validation loss and high validation accuracy. A widening gap in which training loss falls while validation loss rises suggests overfitting. More layers can learn richer patterns, but can also overfit or be harder to optimize. LSTM and GRU gates can help preserve context compared with a simple RNN, but improvement must be measured rather than assumed.

After choosing an architecture and checkpoint using validation results, a separate final test evaluation can measure its performance. The test set here was already inspected in the earlier LSTM experiment, so it is not a completely untouched benchmark across the whole project. Avoid using it to rank these nine experiments. Repeat the experiment with other random seeds before claiming that small differences are reliable.

Implementation reference: [Keras recurrent layers](https://keras.io/api/layers/recurrent_layers/).

