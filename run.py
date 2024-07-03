__author__ = 'yunbo'

import os
import shutil
import argparse
import numpy as np
import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrnn_v2')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# reverse scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
parser.add_argument('--r_sampling_step_1', type=float, default=25000)
parser.add_argument('--r_sampling_step_2', type=int, default=50000)
parser.add_argument('--r_exp_alpha', type=int, default=5000)
# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=1000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=500)
parser.add_argument('--snapshot_interval', type=int, default=500)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

# visualization of memory decoupling
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# action-based predrnn
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')

parser.add_argument('--use_lr_scheduler', type=int, default=0)
parser.add_argument('--rotate', type=int, default=0)
parser.add_argument('--random_flip', type=int, default=0)
parser.add_argument('--blur', type=int, default=0)
parser.add_argument('--affine', type=int, default=0)
parser.add_argument('--resized_crop', type=int, default=0)
parser.add_argument('--random_perspective', type=int, default=0)

args = parser.parse_args()

# -----------------------------------------------------------------------------
def reserve_schedule_sampling_exp(itr):
    r_eta = 0.5
    if itr < args.r_sampling_step_1:
        eta = 1.0
    elif itr < args.r_sampling_step_2:
        eta = 1.0 - (1.0 - r_eta) * (itr - args.r_sampling_step_1) / (args.r_sampling_step_2 - args.r_sampling_step_1)
    else:
        eta = r_eta
    random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size, args.img_width // args.patch_size, args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size, args.img_width // args.patch_size, args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag, (args.batch_size, args.total_length - args.input_length - 1,
                                                   args.img_width // args.patch_size, args.img_width // args.patch_size,
                                                   args.patch_size ** 2 * args.img_channel))
    return real_input_flag

def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size, args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size, args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 1.0, zeros
    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size, args.img_width // args.patch_size, args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag, (args.batch_size, args.total_length - args.input_length - 1,
                                                   args.img_width // args.patch_size, args.img_width // args.patch_size,
                                                   args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag

def train_wrapper(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)

    augmentations = {
        'rotate': args.rotate,
        'random_flip': args.random_flip,
        'random_perspective': args.random_perspective,
        'affine': args.affine,
        'blur': args.blur,
        'resized_crop': args.resized_crop
    }
    if all(value == 0 for value in augmentations.values()):
        augmentations = None

    train_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=True, augmentations=augmentations)

    eta = args.sampling_start_value

    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, args.patch_size)

        if args.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(itr)
        else:
            eta, real_input_flag = schedule_sampling(eta, itr)

        trainer.train(model, ims, real_input_flag, args, itr)

        if itr % args.snapshot_interval == 0:
            model.save(itr)

        if itr % args.test_interval == 0:
            trainer.test(model, test_input_handle, args, itr)

        train_input_handle.next()

def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')

# Define the hyperparameter space
space = {
    'lr': hp.loguniform('lr', -10, -2),
    'num_hidden': hp.choice('num_hidden', ['64,64,64,64', '128,128,128,128']),
    'filter_size': hp.choice('filter_size', [3, 5]),
    'stride': hp.choice('stride', [1, 2])
}

def objective(params):
    # Update args with the current hyperparameters
    args.lr = params['lr']
    args.num_hidden = params['num_hidden']
    args.filter_size = params['filter_size']
    args.stride = params['stride']

    print(f"Current Hyperparameters: {params}")

    # Initialize model
    model = Model(args)

    # Load training data
    train_input_handle, _ = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=True)

    eta = args.sampling_start_value

    total_cost = 0
    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims = train_input_handle.get_batch()
        ims = preprocess.reshape_patch(ims, args.patch_size)

        if args.reverse_scheduled_sampling == 1:
            real_input_flag = reserve_schedule_sampling_exp(itr)
        else:
            eta, real_input_flag = schedule_sampling(eta, itr)

        cost = trainer.train(model, ims, real_input_flag, args, itr)
        total_cost += cost

        train_input_handle.next()

    avg_cost = total_cost / args.max_iterations
    return {'loss': avg_cost, 'status': STATUS_OK}


if __name__ == '__main__':
    # Initialize trials object to store results
    trials = Trials()

    # Run the optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,  # Number of evaluations
        trials=trials
    )

    print(f"Best hyperparameters: {best}")
