import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import pandas as pd
'''
if len(sys.argv) > 1:
	factor = int(sys.argv[1])
else:
	factor = 1
'''
factor = 1
print(f'Using factor {factor}')


arr_list = []
metric_list = []
for fi in sys.argv[1:]:
    arr_list.append(np.loadtxt(fi))
    metric_list.append(fi.rstrip('.txt.'))

arr = np.stack(arr_list)

n = arr.shape[1]
leftover = n % factor
if leftover > 0:
	arr = arr[:, :-leftover]

n = arr.shape[1]

plot_arr = np.reshape(arr, [arr.shape[0], factor, -1]).mean(axis=1)

rg = range(0, n, factor)
sns.lineplot(x=rg, y=plot_arr[0])

plt.savefig('history.png')
