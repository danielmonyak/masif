import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

y_true = np.loadtxt('y_true.txt')
y_pred_probs = np.loadtxt('y_pred_probs.txt')
y_pred = np.loadtxt('y_pred.txt')

# Binarize the output
y_true_bin = label_binarize(y_true, classes=list(range(7)))
n_classes = y.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

