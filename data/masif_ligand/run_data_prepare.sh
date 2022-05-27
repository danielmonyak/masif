./schedule_data_prepare.sh &
disown -h $!
echo $! > pid.txt
