job_name=schedule_no_wait

./$job_name.sh > $job_name.out 2>&1 &
disown -h $!
echo $! > $job_name.txt
