import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

loss = np.loadtxt('loss.txt')

rg = np.arange(0, len(loss))
sns.lineplot(x=rg, y=loss)
#sns.regplot(x=rg, y=loss)

plt.savefig('history.png')
