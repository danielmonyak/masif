import sys
import os

continue_key = input('Continue training from checkpoint? (y/[n]): ')
if (continue_key == '') or (continue_key == 'n'):
    continue_training = False
elif continue_key == 'y':
    continue_training = True
else:
    sys.exit('Please enter a valid choice...')

if continue_training:
    ckpPath = os.path.join('kerasModel', 'ckp')
    if os.path.exists(ckpPath):
        ckpKey = input(f'Use checkpoint at "{ckpPath}"? ([y]/n): ')
        if (ckpKey != '') and (ckpKey != 'y'):
            if ckpKey == 'n':
                ckpPath = input('Enter checkpoint path: ')
            else:
                sys.exit('Please enter a valid choice...')
    else:
        ckpPath = input('Enter checkpoint path: ')
    
    starting_epoch = int(input('Starting epoch: '))
    if starting_epoch < 0:
        sys.exit('Please enter a valid number 0 or greater...')
else:
    ckpPath = os.path.join('kerasModel', 'ckp')
    starting_epoch = 0

num_epochs = int(input('Enter the number of epochs to train for: '))
if num_epochs <= starting_epoch:
    sys.exit(f'Please enter a valid number greater than {starting_epoch}...')

with open('train_vars.py', 'w') as f:
    f.write(f'train_vars["continue_training"] = {continue_training}\n')
    f.write(f'train_vars["ckpPath"] = "{ckpPath}"\n')
    f.write(f'train_vars["starting_epoch"] = {starting_epoch}\n')
    f.write(f'train_vars["num_epochs"] = {num_epochs}\n')
