./schedule_2.sh > schedule_2.out 2>&1 &
disown -h $!
echo $! > schedule_2_pid.txt
