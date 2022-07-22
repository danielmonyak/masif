import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

if len(sys.argv) > 1:
	factor = int(sys.argv[1])
else:
	factor = 1

print(f'Using factor {factor}')

F1 = np.loadtxt('F1.txt')
loss = np.loadtxt('loss.txt')
arr = np.stack([F1, loss])

n = arr.shape[1]
leftover = n % factor
if leftover > 0:
	arr = arr[:, :-leftover]

n = arr.shape[1]

plot_arr = np.reshape(arr, [arr.shape[0], factor, -1]).mean(axis=1)

rg = range(0, n, factor)
sns.lineplot(x=rg, y=plot_arr[0])
sns.lineplot(x=rg, y=plot_arr[1])

plt.savefig('history.png')
