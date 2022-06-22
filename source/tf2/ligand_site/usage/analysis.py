import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dir = 'results'

arr_list = ['lig_true', 'lig_pred', 'recall', 'precision']

input_dict = {}
for arr in arr_list:
  path = os.path.join(dir, arr + '.txt')
  input_dict[arr] = np.loadtxt(path, dtype=float)

df = pd.DataFrame(input_dict)

balanced_acc = balanced_accuracy_score(df['lig_true'], df['lig_pred'])
print('Balanced accuracy: ', round(balanced_acc, 2))

df['correct'] = df['lig_true'] == df['lig_pred']
sns.scatterplot(data = df, x = 'precision', y = 'recall', hue = 'correct')
plt.savefig('plot.png')
