import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

if len(sys.argv) > 1:
	factor = int(sys.argv[1])
else:
	factor = 1

print(f'Using factor {factor}')

loss = np.loadtxt('loss.txt')
auc = np.loadtxt('auc.txt')

rg = range(0, len(loss), factor)
sns.lineplot(x=rg, y=loss)
sns.lineplot(x=rg, y=auc)

plt.savefig('history.png')
