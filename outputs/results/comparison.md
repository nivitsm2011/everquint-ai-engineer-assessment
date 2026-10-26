# Baseline vs residual CNN

Both models used identical settings: seed 42, 10 epochs, batch size 128, SGD (lr 0.1, momentum 0.9, weight decay 0.0005), onecycle schedule, dropout 0.3, gradient clipping off, early-stopping patience 5, and the same augmentation and train/validation split. Training time is the summed epoch time (train + validation) on a shared CPU, so it is approximate.

## Summary

| Metric | Baseline CNN | ResNet CNN | Difference |
|---|---|---|---|
| Parameters | 288,746 | 308,650 | +19,904 |
| Epochs trained | 10 | 10 | - |
| Training time (min) | 21.7 | 81.2 | +59.5 |
| Best validation accuracy | 82.92% (epoch 10) | 75.70% (epoch 10) | -7.22 pp |
| Final train accuracy | 78.70% | 71.20% | -7.50 pp |
| **Test accuracy** | **81.70%** | **75.32%** | **-6.38 pp** |
| Test loss | 0.5302 | 0.6979 | +0.1677 |

## Per-class test accuracy

| Class | Baseline CNN | ResNet CNN | Difference |
|---|---|---|---|
| airplane | 83.6% | 75.6% | -8.0 pp |
| automobile | 92.0% | 88.8% | -3.2 pp |
| bird | 68.8% | 58.5% | -10.3 pp |
| cat | 62.9% | 57.0% | -5.9 pp |
| deer | 79.6% | 69.0% | -10.6 pp |
| dog | 76.8% | 66.1% | -10.7 pp |
| frog | 87.5% | 83.3% | -4.2 pp |
| horse | 83.3% | 75.2% | -8.1 pp |
| ship | 91.8% | 91.4% | -0.4 pp |
| truck | 90.7% | 88.3% | -2.4 pp |

![Validation curves](../plots/comparison_curves.png)

## Conclusion

The residual network did **not** help: test accuracy fell from 81.70% to 75.32% (-6.38 pp) and test loss went from 0.5302 to 0.6979.

- Per class, the most favourable change was **ship** (-0.4 pp) and the least favourable was **dog** (-10.7 pp).
- After epoch 5, validation accuracy was 70.00% for the baseline and 63.74% for the ResNet, which shows how quickly each one trained under the same schedule.
- The train-minus-validation accuracy gap at the last epoch was -4.22 pp (baseline) and -4.50 pp (ResNet); a negative gap means neither model is overfitting.
- Cost: +19,904 parameters (+6.9%) and 3.74x the training time.

At this size (about seven convolution layers) the skip connections did not offset the other architectural changes, so this run gives no evidence that residual connections help here.

Caveats: each model was trained once (seed 42), and the binomial standard error of a test accuracy near 82% is about 0.39 pp, so differences under roughly 0.8 pp are within noise. The ResNet also downsamples with stride-2 convolutions instead of max-pooling, so any difference cannot be attributed to the skip connections alone.
