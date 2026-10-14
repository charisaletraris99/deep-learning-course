# Exercise 2 Residual CNN

## Model used

This model keeps the CNN from Exercise 1 and adds a residual shortcut around
each of its three convolutions. It is not a full ResNet-50. This follows the
assignment requirement to modify the Exercise 1 CNN and then compare the two
models fairly.

The architecture remains:

| Layer | Filters or setting | Output size |
| --- | --- | --- |
| Residual convolution 1 | 32 filters, 3 x 3, stride 1, GELU | 28 x 28 x 32 |
| Max pooling 1 | 2 x 2, stride 2 | 14 x 14 x 32 |
| Residual convolution 2 | 64 filters, 3 x 3, stride 1, GELU | 14 x 14 x 64 |
| Max pooling 2 | 2 x 2, stride 2 | 7 x 7 x 64 |
| Residual convolution 3 | 64 filters, 3 x 3, stride 1, GELU | 7 x 7 x 64 |
| Classifier | Flatten, Dense 128, GELU, dropout 0.3 | 10 probabilities |

## Residual connection

Each residual block calculates:

    output = GELU(convolution(input) + shortcut(input))

The shortcut carries the input around the convolution. If the convolution has
more channels, zero channels are appended to the shortcut before addition:

- Block 1 changes 1 channel to 32 channels, so 31 zero channels are added.
- Block 2 changes 32 channels to 64 channels, so 32 zero channels are added.
- Block 3 already has 64 channels, so it uses the input unchanged.

The shortcut has no trainable weights. This gives gradients a more direct path
to earlier layers, which can make training easier. It does not guarantee
better accuracy or prevent overfitting.

## What stays the same as Exercise 1

The two models use the same data split, 3 x 3 kernels, filters 32, 64, 64,
replicate edge padding, GELU, two 2 x 2 max-pooling layers, Dense 128,
dropout 0.3, Adam learning rate 0.001, 40 epochs, batch size 128, and seed 42.

This makes the comparison focus on the residual connections. The residual
shortcuts add no trainable parameters, so both models have 458,570 trainable
parameters.

## Run and outputs

Run from the repository root:

    .\DL_venv\Scripts\python.exe "assignment-2/excercise 2/train_residual.py"

The new results are written to
assignment-2/outputs/exercise_2_residual_cnn_gelu/. The script creates only
learning_curves.png and confusion_matrix.png. Use the new graphs and terminal
metrics to replace the previous ResNet-50 results in the report.
