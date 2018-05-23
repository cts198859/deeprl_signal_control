import argparse
import os
import subprocess


def parse_args():
    default_train_dir = '/tests_may20'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_train_dir, help="training dir")
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = args.base_dir
    tokens = train_dir.split('/')
    train_folder = tokens[-1].split('_')
    train_folder[0] = 'eval'
    tokens[-1] = '_'.join(train_folder) 
    eval_dir = '/'.join(tokens)
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    folders = os.listdir(train_dir)
    for sc
