
import numpy as np
import os
import pandas as pd

df_list = []
csv_dir = 'CSVs'
csv_files = os.listdir(csv_dir)
for fi in csv_files:
    df_temp = pd.read_csv(os.path.join(csv_dir, fi), header=1, usecols=['Entry ID', 'Sequence Cluster  ID', 'Sequence Cluster Identity Threshold'])
    df_temp = df_temp.fillna(method='ffill', axis=0)
    df_temp = df_temp.loc[df_temp['Sequence Cluster Identity Threshold'] == 30]
    df_list.append(df_temp)

df = pd.concat(df_list)
df = df.rename(columns={'Entry ID':'pdb', 'Sequence Cluster  ID':'cluster'}).convert_dtypes()
uniq_ids = df['pdb'].unique().astype(str)

np.savetxt('all_pdbs.txt', uniq_ids, fmt='%s')

np.random.shuffle(uniq_ids)

using_clusters = []
using_pdbs = []
n_pdbs = len(uniq_ids)
for i, pdb_id in enumerate(uniq_ids):
    if i % 100 == 0:
        print(f'{i} of {n_pdbs} done')
    cur_clusters = df.loc[df['pdb'] == pdb_id, 'cluster']
    if len(np.intersect1d(cur_clusters, using_clusters)) == 0:
        using_clusters.extend(cur_clusters)
        using_pdbs.append(pdb_id)

np.savetxt('filtered_pdbs.txt', using_pdbs, fmt='%s')
