import sys
import os
import pickle

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

        
var_names = ['continue_training', 'ckpPath', 'starting_iteration', 'num_iterations']
train_vars = dict(zip(var_names, [eval(v) for v in var_names]))
with open('train_vars.pickle', 'wb') as handle:
    pickle.dump(train_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)
