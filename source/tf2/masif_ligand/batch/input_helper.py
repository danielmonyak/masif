import sys
import os

continue_key = input('Continue training from checkpoint? (y/[n]): ')
if (continue_key == '') or (continue_key == 'n'):
    continue_training = False
elif continue_key == 'y':
    continue_training = True
else:
    sys.exit('Please enter a valid choice...')

ckpPath = os.path.join('kerasModel', 'ckp')
starting_iteration = 0
if continue_training:
    if os.path.exists(ckpPath + '.index'):
        ckpKey = input(f'Use checkpoint at "{ckpPath}"? ([y]/n): ')
        if (ckpKey != '') and (ckpKey != 'y'):
            if ckpKey == 'n':
                ckpPath = input('Enter checkpoint path: ')
            else:
                sys.exit('Please enter a valid choice...')
    else:
        ckpPath = input('Enter checkpoint path: ')
    
    starting_iteration_key = input(f'Starting iteration [{starting_iteration}]: ')
    if starting_iteration_key != '':
        try:
            starting_iteration = int(starting_iteration_key)
        except:
            sys.exit('Please enter a valid number 0 or greater...')
        if starting_iteration < 0:
            sys.exit('Please enter a valid number 0 or greater...')
    

num_iterations = 10**7
num_iterations_key = input(f'Enter the number of iterations to train for [{num_iterations}]: ')
if num_iterations_key != '':
    try:
        num_iterations = int(num_iterations_key)
    except:
        sys.exit(f'Please enter a valid number greater than {starting_iteration}...')
    if num_iterations <= starting_iteration:
        sys.exit(f'Please enter a valid number greater than {starting_iteration}...')

with open('train_vars.py', 'w') as f:
    f.write('train_vars = {}\n')
    f.write(f'train_vars["continue_training"] = {continue_training}\n')
    f.write(f'train_vars["ckpPath"] = "{ckpPath}"\n')
    f.write(f'train_vars["starting_iteration"] = {starting_iteration}\n')
    f.write(f'train_vars["num_iterations"] = {num_iterations}\n')
