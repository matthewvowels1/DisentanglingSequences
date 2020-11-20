from parameters import get_parameters
from train_test import Trainer_Tester
from torch.backends import cudnn
from utilities import make_folder
import torch
import os

# tensorboard --logdir=./tensorboard_runs

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    make_folder(config.model_save_path, config.RUN)
    make_folder(config.image_path, config.RUN)
    make_folder(config.log_dir, config.RUN)
    make_folder(config.code_path, config.RUN)

    print(type(config))
    with open(os.path.join(config.image_path, config.RUN, 'config.txt'), 'w') as file:
        file.write(str(vars(config)))
    with open(os.path.join(config.model_save_path, config.RUN, 'config.txt'), 'w') as file:
        file.write(str(vars(config)))

    trainer_tester = Trainer_Tester(config)
    if config.train_VDSMSeq or config.train_VDSMEncDec:
        print('Training...')
        trainer_tester.train()
    else:
        print('Testing...')
        trainer_tester.test()



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    config = get_parameters()
    print(config)
    main(config)