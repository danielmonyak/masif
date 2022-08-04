pcregrep -o2 "(Loss *\-* )([0-9.]*)" train_model.out > loss.txt
pcregrep -o2 "(Loss.*Accuracy *\-* )([0-9.]*)" train_model.out > acc.txt
