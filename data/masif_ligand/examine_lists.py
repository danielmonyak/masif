import numpy as np

test = np.load("lists/test_pdbs_sequence.npy").astype(str)
train = np.load("lists/train_pdbs_sequence.npy").astype(str)
val = np.load("lists/val_pdbs_sequence.npy").astype(str)

print("test: ", test.size)
print("train: ", train.size)
print("val: ", val.size)

print("\ntotal: ", test.size + train.size + val.size)
