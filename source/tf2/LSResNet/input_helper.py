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
    ckpKey = input(f'Using checkpoint at {ckpPath}? ([y]/n): ')
    if (ckpKey != '') and (ckpKey != 'y'):
        if ckpKey == 'n':
            ckpPath = input('Enter checkpoint path: ')
        else:
            sys.exit('Please enter a valid choice...')
    
    starting_epoch = int(input('Starting epoch: '))
else:
    ckpPath = os.path.join('kerasModel', 'ckp')
    starting_epoch = 0

num_epochs = int(input('Enter the number of epochs to train for: '))

