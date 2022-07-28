import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#loss_most = np.loadtxt('loss_most.txt')
loss = np.loadtxt('loss.txt')

#loss = np.concatenate([loss_most, loss])

rg = range(0, loss.shape[0])
sns.lineplot(x=rg, y=loss)

plt.savefig('history.png')
