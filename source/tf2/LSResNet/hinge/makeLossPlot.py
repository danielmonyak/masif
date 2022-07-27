import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

loss = np.loadtxt('loss.txt')
rg = range(0, loss.shape[0])
sns.lineplot(x=rg, y=loss)

plt.savefig('history.png')
