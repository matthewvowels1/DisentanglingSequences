import argparse
import os
def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # training settings
    parser.add_argument('--RUN', type=str, default='001')
    parser.add_argument('--test_harness', type=str2bool, default=False)
    parser.add_argument('--bs', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bs_per_epoch', type=int, default=2)  # 50
    parser.add_argument('--num_test_ids', type=int, default=1)  # 'batch size' for testing swaps and sequence gen
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='MUG-FED', choices=['MUG-FED'])
    parser.add_argument('--pretrained_model_VDSMEncDec', type=int, default=None)
    parser.add_argument('--pretrained_model_VDSMSeq', type=int, default=None)
    parser.add_argument('--train_VDSMEncDec', type=str2bool, default=False)
    parser.add_argument('--train_VDSMSeq', type=str2bool, default=False)
    parser.add_argument('--tboard_log', type=str2bool, default=True)
    parser.add_argument('--model_test_interval', type=int, default=1)
    parser.add_argument('--model_save_interval', type=int, default=2)
    parser.add_argument('--image_path', type=str, default='./images/')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--log_dir', type=str, default='./tensorboard_runs/')
    parser.add_argument('--code_path', type=str, default='./codes/')

    # Blended model hyper-parameters
    parser.add_argument('--likelihood', type=str, default='Bernoulli', choices=['Bernoulli', 'Normal', 'Laplace'])
    parser.add_argument('--z_dim', type=int, default=20)
    parser.add_argument('--n_e_w', type=int, default=15)
    parser.add_argument('--temp_id_end', type=float, default=1)
    parser.add_argument('--temp_id_start', type=float, default=2.0)
    parser.add_argument('--temp_id_frac', type=int, default=1)  # if this is 3, then frac is 1/3
    parser.add_argument('--anneal_end_id', type=float, default=0.8)
    parser.add_argument('--anneal_start_id', type=float, default=0.1)
    parser.add_argument('--anneal_frac_id', type=int, default=2)  # if this is 3, then frac is 1/3
    parser.add_argument('--anneal_start_t', type=float, default=10)   # these plot out a 4 point cubic spline
    parser.add_argument('--anneal_mid_t1', type=float, default=0.1)
    parser.add_argument('--anneal_mid_t2', type=float, default=0.1)
    parser.add_argument('--anneal_end_t', type=float, default=1)
    parser.add_argument('--anneal_t_midfrac1', type=float, default=0.3)
    parser.add_argument('--anneal_t_midfrac2', type=float, default=0.6)
    parser.add_argument('--T_0_frac', type=int, default=6)
    parser.add_argument('--T_mult', type=int, default=1)
    parser.add_argument('--lr_VDSMEncDec', type=float, default=0.0005)

    # Sequence model params
    parser.add_argument('--dynamics_dim', type=int, default=10)
    parser.add_argument('--test_temp_id', type=float, default=10)
    parser.add_argument('--trans_dim', type=int, default=20)
    parser.add_argument('--rnn_dim', type=int, default=512)
    parser.add_argument('--rnn_dropout', type=float, default=0.0)
    parser.add_argument('--rnn_layers', type=int, default=3)
    parser.add_argument('--lr_VDSMSeq', type=float, default=0.0005)
    parser.add_argument('--lr_resume', type=float, default=None)
    parser.add_argument('--anneal_end_dynamics', type=float, default=1)
    parser.add_argument('--anneal_start_dynamics', type=float, default=0.01)
    parser.add_argument('--anneal_frac_dynamics', type=int, default=1)  # if this is 3, then frac is 1/3

    config = parser.parse_args()

    return config