import numpy as np

test = np.load("lists/test_pdbs_sequence.npy").astype(str)
train = np.load("lists/train_pdbs_sequence.npy").astype(str)
val = np.load("lists/val_pdbs_sequence.npy").astype(str)

print("test: ", test.size)
print("train: ", train.size)
print("val: ", val.size)

print("\ntotal: ", test.size + train.size + val.size)
'''
#pdb='1GSA_ACBD'
pdb='1U5U_A'
print(pdb in test)
print(pdb in train)
print(pdb in val)
#np.savetxt('test.txt', test, delimiter=" ", fmt="%s")
'''
