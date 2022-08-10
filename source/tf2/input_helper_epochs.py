import sys
import os
import pickle

continue_dict = {'' : False, 'n' : False, 'y' : True}
continue_key = input('Continue training from checkpoint? (y/[n]): ')
try:
    continue_training = continue_dict[continue_key]
except:
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
    

num_epochs = 100
num_epochs_key = input(f'Enter the number of epochs to train for [{num_epochs}]: ')
if num_epochs_key != '':
    try:
        num_epochs = int(num_epochs_key)
    except:
        sys.exit(f'Please enter a valid number greater than {starting_epoch}...')
    if num_epochs <= starting_epoch:
        sys.exit(f'Please enter a valid number greater than {starting_epoch}...')

lr_default_str = '1e-3'
lr_key = input(f'Learning rate [{lr_default_str}]: ')
if lr_key == '':
    lr = float(lr_default_str)
else:
    try:
        lr = float(lr_key)
    except:
        sys.exit(f'Please enter a valid number...')
        
var_names = ['continue_training', 'ckpPath', 'starting_epoch', 'num_epochs', 'lr']
train_vars = dict(zip(var_names, [eval(v) for v in var_names]))
with open('train_vars.pickle', 'wb') as handle:
    pickle.dump(train_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)
