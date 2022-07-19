pcregrep -o2 "( loss\: )([\.0-9]+)" train_individually.out > loss.txt
pcregrep -o2 "( auc\: )([\.0-9]+)" train_individually.out > auc.txt
