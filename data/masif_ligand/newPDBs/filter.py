import numpy as np
import os
import pandas as pd
import sys

mode = sys.argv[1]

df = pd.read_csv('df.csv', index_col=0)
all_pdbs = np.loadtxt('all_pdbs.txt', dtype=str)

cut = lambda x : np.char.partition(x, '_')[:,0]

if mode == 'reg':
    old_pdbs = np.loadtxt('/home/daniel.monyak/software/masif/data/masif_ligand/lists/sequence_split_list_UNIQUE.txt', dtype=str)
    using_pdbs = old_pdbs[np.isin(cut(old_pdbs), all_pdbs)].tolist()
    using_clusters = df['cluster'][np.isin(df.index, cut(using_pdbs))].tolist()
elif mode == 'solvents':
    using_pdbs = np.loadtxt('reg_filtered_pdbs.txt', dtype=str).tolist()
    using_clusters = np.loadtxt('used_clusters.txt', dtype=str).tolist()
else:
    sys.exit('Enter either "reg" or "solvents"...')

uniq_ids = all_pdbs[~np.isin(all_pdbs, cut(using_pdbs))]
np.random.shuffle(uniq_ids)

n_pdbs = len(uniq_ids)
for i, pdb_id in enumerate(uniq_ids):
    if i % 1000 == 0:
        print(f'{i} of {n_pdbs} done')
    cur_clusters = df.loc[pdb_id, 'cluster']
    if type(cur_clusters) in [int, np.int32, np.int64]:
        cur_clusters = [cur_clusters]
    if len(np.intersect1d(cur_clusters, using_clusters)) == 0:
        using_clusters.extend(cur_clusters)
        using_pdbs.append(pdb_id + f'_{df.loc[pdb_id, "chains"][0]}_')

np.savetxt('filtered_pdbs.txt', using_pdbs, fmt='%s')
np.savetxt('used_clusters.txt', using_clusters, fmt='%s')

print('Finished!')
