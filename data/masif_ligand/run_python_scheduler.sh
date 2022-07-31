python schedule_python.py > schedule_python.out 2>&1 &
disown -h $!
echo $! > schedule_python_pid.txt
