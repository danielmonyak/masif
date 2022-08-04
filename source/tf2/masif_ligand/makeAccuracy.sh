pcregrep -o2 "( sparse_categorical_accuracy\: )([\.0-9]+)" train_model.out > accuracy.txt
pcregrep -o2 "( val_sparse_categorical_accuracy\: )([\.0-9]+)" train_model.out > val_accuracy.txt
