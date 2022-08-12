import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

fi = sys.argv[1]
arr = np.loadtxt(fi))

n = arr.shape[1]

rg = range(0, n)
sns.lineplot(x=rg, y=arr)

plt.savefig('history.png')
