pid=$1
while ps -o pid ax | grep -q $pid; do
	sleep 10
done

for i in {1..5}; do
	tput bel
	sleep 2
done
