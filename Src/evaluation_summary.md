# Model Evaluation and Training Summary

This report displays the training, validation (at the best epoch), and test metrics for the LViT model across all 4 settings.

| Dataset / Setting | Best Epoch | Train Loss | Train IoU | Train Dice | Val Loss | Val IoU | Val Dice | Test Dice | Test IoU |
|---|---|---|---|---|---|---|---|---|---|
| **Covid19** (Supervised) | 79 | 0.0960 | 0.7753 | 0.8632 | 0.1572 | 0.6981 | 0.7944 | 0.8127 | 0.7246 |
| **Covid19** (Semi_supervised) | 30 | 0.2432 | 0.7714 | 0.8591 | 0.2028 | 0.6378 | 0.7528 | 0.7672 | 0.6693 |
| **MosMedDataPlus** (Supervised) | 88 | 0.1194 | 0.6783 | 0.7973 | None | None | 0.6720 | 0.1339 | 0.1100 |
| **MosMedDataPlus** (Semi_supervised) | 88 | 0.2795 | 0.6713 | 0.7879 | None | None | 0.6386 | 0.1292 | 0.1044 |
