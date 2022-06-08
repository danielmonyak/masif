import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

y_true = np.loadtxt('y_true.txt')
y_pred_probs = np.loadtxt('y_pred_probs.txt')
y_pred = np.loadtxt('y_pred.txt')

balanced_acc = balanced_accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class = 'ovr', labels = np.arange(7))

print('Balanced accuracy: ', round(balanced_acc, 2))
print('ROC AUC score: ', round(roc_auc, 2))
