import numpy as np
import os
import pandas as pd
import sys

df_list = []
csv_dir = sys.argv[1]

csv_files = os.listdir(csv_dir)
for fi in csv_files:
    df_temp = pd.read_csv(os.path.join(csv_dir, fi), header=1, usecols=['Entry ID', 'Sequence Cluster  ID', 'Sequence Cluster Identity Threshold', 'Asym ID'])
    df_list.append(df_temp)

df = pd.concat(df_list)
df = df.rename(columns={'Entry ID':'pdb', 'Sequence Cluster  ID':'cluster', 'Sequence Cluster Identity Threshold':'thresh'}).convert_dtypes()

df['pdb'] = df['pdb'].fillna(method='ffill', axis=0)
df['Asym ID'][df['thresh'] != 100] = ''
df['Asym ID'] = df['Asym ID'].fillna('')
df['chains'] = df.groupby('pdb')['Asym ID'].transform(lambda x: ''.join(x))

df = df.loc[df['thresh'] == 30]
df = df.set_index('pdb')[['cluster', 'chains']]

counts = df.index.value_counts()
df = df.loc[counts.index[counts < 5]]

len_arr = df['chains'].str.len()
df = df.loc[len_arr <= 4]

uniq_ids = df.index.unique().astype(str)

np.savetxt('all_pdbs.txt', uniq_ids, fmt='%s')
df.to_csv('df.csv')
