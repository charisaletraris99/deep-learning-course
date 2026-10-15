# Assignment 3 Exercise 1 Question 4

`imdb_lstm.py` trains an LSTM to predict whether an IMDB movie review is negative (0) or positive (1). An LSTM is a type of recurrent neural network. The assignment allows other RNN architectures, so this uses an LSTM rather than a SimpleRNN layer.

## Run from the repository folder

```powershell
.\DL_venv\Scripts\python.exe .\assigment-3\imdb_lstm.py
```

The default is 35 epochs, with no early stopping so you can observe the full learning and overfitting trends. To change the number, edit `num_epochs = 35` in the script. The first run downloads IMDB if it is not already cached. Training on a CPU can take a while because every review has up to 500 sequence steps.

Results go to `results_lstm_70_10_20`. A later run replaces those output files. Dependencies are TensorFlow, NumPy, Matplotlib and scikit-learn, already present in `DL_venv`.

## What the model does

1. **Embedding:** turns each word ID into a vector of 32 learned numbers. The vocabulary is limited to 10,000 IDs.
2. **LSTM:** reads the vectors in order and keeps useful information in a memory state. It has 32 units.
3. **Dense with sigmoid:** produces a probability of positive sentiment. A probability of at least 0.5 gives label 1; otherwise it gives label 0.

For example, the word "not" can change the meaning of words that appear later. A simple RNN may struggle to learn to retain this information across a long review because its gradient can become very small. An LSTM has a memory cell and gates that learn what to keep, forget and output. This helps information and learning signals travel across more sequence steps. It reduces the vanishing-gradient difficulty; it does not eliminate it or guarantee higher accuracy.

A SimpleRNN has fewer parameters and is generally cheaper per step. A GRU is another gated RNN with fewer gates and often fewer parameters than an LSTM. We use one small LSTM layer as a straightforward way to handle long reviews without adding a complex architecture. We have not run a comparison experiment, so we cannot claim this particular LSTM outperforms those alternatives.

## Training and evaluation

- All 50,000 reviews are combined and randomly split into 35,000 training (70%), 5,000 validation (10%) and 10,000 test (20%) reviews. Both splitting steps preserve the positive/negative balance and use seed 42. The second split takes 12.5% of the remaining 40,000 reviews for validation. The new sets do not overlap. This replaces the official IMDB split; the vocabulary IDs remain those supplied by Keras rather than being rebuilt from the new training set.
- Reviews retain their last 500 tokens. Short reviews receive zero padding on the right. `mask_zero=True` tells the LSTM to ignore padding. Right padding supports the optimized LSTM GPU path when a compatible GPU is available.
- Adam updates the model weights using binary cross-entropy, which measures the error in predicted probabilities. Gradient clipping limits excessively large gradients; the LSTM gates address the separate problem of vanishing gradients.
- Only training reviews update the weights. The number of epochs is fixed; no test score controls training or model selection.
- After each epoch, all three sets are evaluated with the same current weights. This makes the three curves comparable. The training curve therefore shows end-of-epoch evaluation, not the average of changing weights during training.
- Test curves are included because they were requested. Use validation results, not test results, for any later tuning; repeatedly choosing changes based on the test curves would compromise the test set as an independent evaluation.

## Output files

- `accuracy.png`: train, validation and test accuracy by epoch. Higher is better.
- `loss.png`: train, validation and test binary cross-entropy by epoch. Lower is better.
- `confusion_matrix.png`: final test predictions, with actual classes as rows and predicted classes as columns. Correct predictions are on the diagonal.
- `classification_report.txt`: precision, recall, F1 score and accuracy on the final test predictions.
- `history.json`: the exact values behind the learning curves.
- `model.keras`: the final trained model, including its embedding weights.

The dashed line marks the epoch with the lowest validation loss, not an automatic diagnosis of the exact overfitting onset. Look for a sustained rise in validation loss while training loss continues falling; one noisy epoch is not enough. All 35 epochs still run, and the saved model is the final model. If training accuracy rises while validation loss rises, the model may be overfitting. If all accuracies remain near 0.5 on this balanced task, the model is learning little beyond guessing. The confusion matrix is an additional diagnostic, not an extra graph explicitly required by the assignment.

## Documentation

- [Keras LSTM layer](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Keras Embedding layer and padding mask](https://keras.io/api/layers/core_layers/embedding/)
