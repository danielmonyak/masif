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
 	./re_precompute.sh $p > $out/$p.out 2>$err/$p.err &
	disown -h $!
	i=$((i+1))
	if [ $i -eq $batchSize ]; then
		sleep $sleep_time
		i=0
	fi
done < todo.txt

echo Finished!
