./schedule_data_prepare.sh &
disown -h $!
echo $! > schedule_data_prepare_pid.txt
