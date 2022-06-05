out=output_files
err=error_files
if [ ! -d $out ]; then mkdir $out; fi
if [ ! -d $err ]; then mkdir $err; fi

i=0
batchSize=8

unset running
while read p; do
	if [ $(( i % $batchSize )) -eq 0 ]; then
		for pid in ${running[@]}; do
			while ps -o pid ax | grep -q $pid; do
				sleep 60
			done
		done
		running=()
	fi
	FIELD1=$(echo $p| cut -d" " -f1)
       	PDBID=$(echo $FIELD1| cut -d"_" -f1)
      	CHAIN1=$(echo $FIELD1| cut -d"_" -f2)
       	CHAIN2=$(echo $FIELD1| cut -d"_" -f3)
 	./data_prepare_one.sh $PDBID\_$CHAIN1\_$CHAIN2 > output_files/$p.out 2>error_files/$p.err &
	disown -h $!
	running+=($!)
	i=$((i+1))
done < lists/sequence_split_list_UNIQUE.txt
