pcregrep -o2 "( categorical_accuracy\: )([\.0-9]+)" train_model.out > accuracy.txt
pcregrep -o2 "( val_categorical_accuracy\: )([\.0-9]+)" train_model.out > val_accuracy.txt
