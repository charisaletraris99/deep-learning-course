# Assignment 4 Charis Aletraris

## Experiment setup

I used the original 32 x 32 CIFAR-100 images with no resizing, a 45,000/5,000/10,000 training/validation/test split, seed 42, batch size 64, 30 epochs and AdamW learning rate 0.0001, weight decay 0.001 and global gradient clip norm 1.0. All models used the same VGG16 channel preprocessing and device /GPU:0. I recorded test metrics after each epoch without using them for weight updates or model selection.

## Architectures

I used the VGG16 convolutional backbone with global average pooling and a 100-class softmax. Its 32 x 32 inputs produce a 1 x 1 x 512 final feature map. The scratch model used random weights; the transfer model used ImageNet weights. All layers were trainable in both models.

My ViT used 8 x 8 patches (16 tokens), 192 features, 6 pre-normalized encoder blocks, 6 attention heads with 32 features per head, GELU feed-forward layers of sizes 768 and 192, dropout 0.1, residual additions, learned positional embeddings, final LayerNorm, mean pooling and a 100-class softmax. I normalized patch projections before adding positions. I selected this compact architecture to limit attention cost and training time. It used no pretrained weights.

## Exercise 1 VGG16 from scratch

### Results for VGG16 from scratch

I obtained final validation accuracy of 41.56% and final test accuracy of 43.78%. The final test loss was 4.2964.

| Epoch | Training accuracy | Validation accuracy | Test accuracy | Training loss | Validation loss | Test loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.59% | 10.98% | 10.49% | 4.0840 | 3.7701 | 3.7693 |
| 15 | 87.67% | 42.14% | 43.97% | 0.3775 | 3.4638 | 3.3231 |
| 30 | 96.22% | 41.56% | 43.78% | 0.1199 | 4.4662 | 4.2964 |

The final training minus validation accuracy gap was +54.66 percentage points. I consider this gap together with the loss curves when assessing overfitting.

![Figure 1](figure_1_vgg16_scratch.png)

Figure 1. Left: accuracy; right: loss. Training: solid blue; validation: dashed orange; test: dash-dot green.

## Exercise 2 VGG16 transfer learning

### Results for VGG16 transfer learning

I obtained final validation accuracy of 1.00% and final test accuracy of 1.00%. The final test loss was 4.6052.

| Epoch | Training accuracy | Validation accuracy | Test accuracy | Training loss | Validation loss | Test loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.86% | 1.00% | 1.00% | 4.6688 | 4.6052 | 4.6052 |
| 15 | 0.89% | 1.00% | 1.00% | 4.6052 | 4.6052 | 4.6052 |
| 30 | 0.86% | 1.00% | 1.00% | 4.6052 | 4.6052 | 4.6052 |

The final training minus validation accuracy gap was -0.14 percentage points. I consider this gap together with the loss curves when assessing overfitting.

![Figure 2](figure_2_vgg16_transfer.png)

Figure 2. Left: accuracy; right: loss. Training: solid blue; validation: dashed orange; test: dash-dot green.

## Exercise 3 ViT from scratch

### Results for ViT from scratch

I obtained final validation accuracy of 28.78% and final test accuracy of 29.84%. The final test loss was 3.4225.

| Epoch | Training accuracy | Validation accuracy | Test accuracy | Training loss | Validation loss | Test loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.07% | 11.30% | 11.69% | 4.0942 | 3.8848 | 3.8677 |
| 15 | 39.36% | 27.96% | 28.50% | 2.3181 | 3.0120 | 2.9533 |
| 30 | 66.54% | 28.78% | 29.84% | 1.1706 | 3.5289 | 3.4225 |

The final training minus validation accuracy gap was +37.76 percentage points. I consider this gap together with the loss curves when assessing overfitting.

![Figure 3](figure_3_vit_scratch.png)

Figure 3. Left: accuracy; right: loss. Training: solid blue; validation: dashed orange; test: dash-dot green.

## Comparison of results

| Model | Parameters | Final validation accuracy | Final test accuracy | Final test loss | Mean epoch seconds |
|---|---:|---:|---:|---:|---:|
| VGG16 from scratch | 14,765,988 | 41.56% | 43.78% | 4.2964 | 119.55 |
| VGG16 transfer learning | 14,765,988 | 1.00% | 1.00% | 4.6052 | 119.26 |
| ViT from scratch | 2,729,380 | 28.78% | 29.84% | 3.4225 | 114.84 |

### Exercise 2 Effect of transfer learning

Transfer learning reduced final test accuracy by 42.78 percentage points. Its final validation accuracy difference was -40.56 percentage points. As discussed in Lecture 5, pretrained visual features can reduce the learning required for a new task. This measured result describes their benefit or limitation under my training settings.

### Exercise 3 Comparison with VGG16 from scratch

Model with the higher final validation accuracy: VGG16 from scratch. ViT minus VGG16 final validation accuracy was -12.78 percentage points. I compare the final epoch rather than the maximum on either curve. Convolution supplies local shared filters, while attention learns relationships between patches (Lectures 3, 7 and 9). The different parameter counts also affect this comparison.

### Learning speed

I compare time and epochs to the preselected target of 20.0% validation accuracy.

- VGG16 from scratch: epoch 3, 6.30 minutes.
- VGG16 transfer learning: not reached within the training budget.
- ViT from scratch: epoch 6, 11.83 minutes.

Model reaching the target in less measured time: VGG16.

Mean epoch times were 119.55 seconds for VGG16 and 114.84 seconds for ViT.
Excluding epoch 1, the means were 118.78 and 113.63 seconds, respectively.

Test evaluation time is recorded separately and excluded from these epoch times. Faster epochs alone do not establish faster learning. My results use one seed and shared hyperparameters, which may not be optimal for each architecture.

![Figure 4](figure_4_test_accuracy_comparison.png)

Figure 4. Test accuracy per epoch for all three models.

![Figure 5](figure_5_test_loss_comparison.png)

Figure 5. Test loss per epoch for all three models.

![Figure 6](figure_6_learning_speed.png)

Figure 6. Validation accuracy against epochs and cumulative training and validation time.

![Figure 7](figure_7_epoch_time.png)

Figure 7. Training and validation time per epoch for VGG16 and ViT from scratch.

Sources of measurements: history.csv, summary.csv and run.json. Conceptual sources: course Lectures 3, 4, 5, 7 and 9.
