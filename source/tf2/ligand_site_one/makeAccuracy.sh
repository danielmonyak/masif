pcregrep -o2 "( auc\: )([\.0-9]+)" train_model.out > auc.txt
pcregrep -o2 "( loss\: )([\.0-9]+)" train_model.out > loss.txt
