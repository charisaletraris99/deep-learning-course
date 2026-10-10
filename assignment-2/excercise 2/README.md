# Exercise 2: ResNet bottleneck connections

## Model selected

This exercise uses the bottleneck ResNet family shown in the lecture image. A
bottleneck block has three convolutions:

input -> 1 x 1 -> 3 x 3 -> 1 x 1 -> add shortcut -> GELU
   |__________________________________________________|

As we see in the lesson, **ResNet-50** uses 3, 4, 6, and 3 bottleneck blocks
in its four stages. These 16 blocks contain 48 convolutions, the small input
stem adds one convolution, and the final classifier is the fiftieth learned
layer.

We keep only ResNet-50. ResNet-101 and ResNet-152 have many more blocks, so
they would take much more time to train for 40 epochs. Fashion-MNIST images
are small (28 x 28), and Exercise 2 needs one clear comparison with Exercise
1. ResNet-50 is already a deep model for this dataset, so it is enough to
study how residual connections affect training.

## Zero padding and matching the shortcut

The assignment asks us to match different dimensions by padding the smaller
shortcut with zeros. This code follows that idea:

- The spatial borders of every convolution use padding="same", which adds
  zero pixels at the image border.
- When a new stage has more channels, the shortcut keeps its existing channels
  and appends zero channels. It does not use a learned 1 x 1 projection.
- At the start of stages 2, 3, and 4, both paths reduce height and width by a
  factor of two. The shortcut uses 2 x 2 average pooling with stride 2 before
  zero channels are appended.

The feature-map sizes in the default ResNet-50 are:

| Part | Output size |
| --- | --- |
| Input | 28 x 28 x 1 |
| Small 3 x 3 stem | 28 x 28 x 16 |
| Stage 1: 3 blocks | 28 x 28 x 64 |
| Stage 2: 4 blocks | 14 x 14 x 128 |
| Stage 3: 6 blocks | 7 x 7 x 256 |
| Stage 4: 3 blocks | 4 x 4 x 512 |
| Global average pooling | 512 values |
| Classifier | 10 probabilities |

The first 1 x 1 convolution reduces the number of channels inside a block. The
3 x 3 convolution learns nearby visual patterns. The final 1 x 1 convolution
expands the channels to the size needed for the addition. This gives the model
depth without making every 3 x 3 convolution unnecessarily expensive.

## Why residual connections help

For a normal stack of layers, backpropagation must pass through every layer.
In a deep network, these gradients can become very small. A residual block
adds a direct shortcut: output = F(input) + shortcut(input).

This gives the gradient an additional direct route to earlier layers. It can
make a very deep model easier to train. It does not guarantee higher test
accuracy, and it does not by itself prevent overfitting.

## Training settings and outputs

The assignment data split remains 54,000 training, 6,000 validation, and
10,000 test images. Training uses 40 epochs, batch size 128, Adam learning
rate 0.001, categorical cross-entropy, GELU activations, and dropout 0.3.
The weights from the epoch with the lowest validation loss are used for the
final test evaluation.

Run from the repository root:

    .\DL_venv\Scripts\python.exe "assignment-2/excercise 2/train_residual.py"

The script writes only learning_curves.png and confusion_matrix.png to
assignment-2/outputs/exercise_2_residual_gelu/. Compare the selected
ResNet-50 validation loss, test loss, test accuracy, and confusion matrix
with Exercise 1.

Reference: He et al., Deep Residual Learning for Image Recognition, CVPR 2016.
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
