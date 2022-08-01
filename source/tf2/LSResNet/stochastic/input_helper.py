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
starting_epoch = 0
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
    
    starting_epoch_key = input(f'Starting epoch [{starting_epoch}]: ')
    if starting_epoch_key != '':
        try:
            starting_epoch = int(starting_epoch_key)
        except:
            sys.exit('Please enter a valid number 0 or greater...')
        if starting_epoch < 0:
            sys.exit('Please enter a valid number 0 or greater...')
    

num_epochs = 10**4
num_epochs_key = input(f'Enter the number of epochs to train for [{num_epochs}]: ')
if num_epochs_key != '':
    try:
        num_epochs = int(num_epochs_key)
    except:
        sys.exit(f'Please enter a valid number greater than {starting_epoch}...')
    if num_epochs <= starting_epoch:
        sys.exit(f'Please enter a valid number greater than {starting_epoch}...')

with open('train_vars.py', 'w') as f:
    f.write('train_vars = {}\n')
    f.write(f'train_vars["continue_training"] = {continue_training}\n')
    f.write(f'train_vars["ckpPath"] = "{ckpPath}"\n')
    f.write(f'train_vars["starting_epoch"] = {starting_epoch}\n')
    f.write(f'train_vars["num_epochs"] = {num_epochs}\n')
