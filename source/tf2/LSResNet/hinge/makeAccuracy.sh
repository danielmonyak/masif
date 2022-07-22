pcregrep -o2 "( F1\: )([\.0-9]+)" train_model.out > F1.txt
pcregrep -o2 "( loss\: )([\.0-9]+)" train_model.out > loss.txt
