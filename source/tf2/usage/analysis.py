import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dir = 'results'

#arr_list = ['pdbs', 'lig_true', 'lig_pred', 'recall', 'precision']
#dtype_list = [str, float, float, float, float]
arr_list = ['lig_true', 'lig_pred', 'recall', 'precision']
dtype_list = [float, float, float, float]

input_dict = {}
for i, arr in enumerate(arr_list):
  path = os.path.join(dir, arr + '.txt')
  input_dict[arr] = np.loadtxt(path, dtype=dtype_list[i])

df = pd.DataFrame(input_dict)

balanced_acc = balanced_accuracy_score(df['lig_true'], df['lig_pred'])
print('Balanced accuracy: ', round(balanced_acc, 2))

df['correct'] = df['lig_true'] == df['lig_pred']
sns.scatterplot(data = df, x = 'precision', y = 'recall', hue = 'correct')
plt.savefig('plot.png')
'''
ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
conf_mat = confusion_matrix(df['lig_true'], df['lig_pred'], normalize = 'true')
disp = ConfusionMatrixDisplay(conf_mat, display_labels = ligands)
disp.plot()
plt.savefig('confusion_matrix.png')
'''
