import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

lig_true_file = os.path.join(outdir, 'lig_true.txt')
lig_pred_file = os.path.join(outdir, 'lig_pred.txt')

lig_true = np.loadtxt(lig_true_file, dtype=int)
lig_pred = np.loadtxt(lig_pred_file, dtype=int)

balanced_acc = balanced_accuracy_score(lig_true, lig_pred)
print('Balanced accuracy: ', round(balanced_acc, 2))
