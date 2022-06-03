import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score

size = 100
y_true = np.random.choice([0, 1], size)
y_pred = np.random.random(size)

print('balanced accuracy: ', balanced_accuracy_score(y_true, y_pred))
