import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

if len(sys.argv) > 1:
	factor = int(sys.argv[1])
else:
	factor = 1

print(f'Using factor {factor}')

acc = np.loadtxt('accuracy.txt')
val_acc = np.loadtxt('val_accuracy.txt')

arr = np.stack([acc, val_acc])
plot_arr = np.reshape(arr, [arr.shape[0], factor, -1]).mean(axis=1)

rg = range(0, len(acc), factor)
sns.lineplot(x=rg, y=plot_arr[0])
sns.lineplot(x=rg, y=plot_arr[1])

plt.savefig('history.png')
