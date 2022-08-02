out=output_files
err=error_files
if [ ! -d $out ]; then mkdir $out; fi
if [ ! -d $err ]; then mkdir $err; fi

batchSize=15
sleep_time=150

i=0
unset running
while read p; do
	echo $p
 	./data_prepare_one.sh $p > $out/$p.out 2>$err/$p.err &
	disown -h $!
	i=$((i+1))
	if [ $i -eq $batchSize ]; then
		sleep $sleep_time
		i=0
	fi
#done < lists/sequence_split_list_UNIQUE.txt
#done < newPDBs/filtered_pdbs.txt
done < newPDBs/todo.txt

echo Finished!
