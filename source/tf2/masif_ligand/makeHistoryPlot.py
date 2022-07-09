import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#loss = np.loadtxt('loss.txt')
#val_loss = np.loadtxt('val_loss.txt')

#rg = range(len(loss))
#sns.lineplot(x=rg, y=loss)
#sns.lineplot(x=rg, y=val_loss)

acc = np.loadtxt('accuracy.txt')
val_acc = np.loadtxt('val_accuracy.txt')

rg = range(len(acc))
sns.lineplot(x=rg, y=acc)
sns.lineplot(x=rg, y=val_acc)

plt.savefig('history.png')
