./schedule_data_prepare.sh > schedule_data_prepare.out 2>&1 &
disown -h $!
echo $! > schedule_data_prepare_pid.txt
