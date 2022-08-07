import numpy as np
import pandas as pd
import os
import sys

if len(sys.argv) > 1:
    outdir = sys.argv[1]
else:
    outdir = 'results'

results = pd.read_csv(os.path.join(outdir, 'results.csv'))
BIG_results = pd.read_csv(os.path.join(outdir, 'BIG_results.csv'), index_col = 0)



BIG_results['precision'] = BIG_results['matched']/BIG_results['n_pockets_pred']
BIG_results['recall'] = BIG_results['matched']/BIG_results['n_pockets_true']

for tup in [('all', results.index, BIG_results.index), ('test', results['dataset']=='test', BIG_results['dataset']=='test')]:
    print(tup[0], '\n')

    temp_results = results.loc[tup[1]]
    temp_BIG_results = BIG_results.loc[tup[2]]

    print('Recall of pockets: ', round(temp_BIG_results['recall'].mean(), 2))
    print('Precision of pocket predictions: ', round(temp_BIG_results['precision'].mean(), 2), '\n\n')

test = results.loc[results['dataset'] == 'test']
