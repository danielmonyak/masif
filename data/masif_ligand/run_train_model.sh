./train_model.slurm > train_model.out 2>train_model.err &
disown -h $!
echo $! > train_model_pid.txt
