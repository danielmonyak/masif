import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = np.loadtxt('y_true.txt')
y_pred = np.loadtxt('y_pred.txt')

ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
conf_mat = confusion_matrix(y_true, y_pred, normalize = 'true')
disp = ConfusionMatrixDisplay(conf_mat, display_labels = ligands)
disp.plot()
plt.savefig('confusion_matrix.png')
