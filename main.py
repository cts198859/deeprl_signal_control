import argparse
import configparser
import numpy as np
import os
import signal
import tensorflow as tf
import threading

from agents.models import A2C
from envs.wrapper import GymEnv
from train import Trainer


def parse_args():
    default_config_path = '/Users/tchu/Documents/Uhana/remote/deeprl_signal_control/simple_config.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=False,
                        default=default_config_path, help="config path")
    return parser.parse_args()


def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')


def init_out_dir(base_dir):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    save_path = base_dir + '/model/'
    log_path = base_dir + '/log/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    return save_path, log_path


def single_gym_env():
    args = parse_args()
    parser = configparser.ConfigParser()
    parser.read(args.config_path)
    seed = parser.getint('TRAIN_CONFIG', 'SEED')
    env_name = parser.get('ENV_CONFIG', 'NAME')
    env = GymEnv(env_name)
    env.seed(seed)
    n_a = env.n_a
    n_s = env.n_s
    total_step = int(parser.getfloat('TRAIN_CONFIG', 'MAX_STEP'))
    base_dir = parser.get('TRAIN_CONFIG', 'BASE_DIR')
    save_step = int(parser.getfloat('TRAIN_CONFIG', 'SAVE_INTERVAL'))
    log_step = int(parser.getfloat('TRAIN_CONFIG', 'LOG_INTERVAL'))
    save_path, log_path = init_out_dir(base_dir)

    tf.set_random_seed(seed)
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess = tf.Session(config=config)
    model = A2C(sess, n_s, n_a, total_step, model_config=parser['MODEL_CONFIG'])
    saver = tf.train.Saver(max_to_keep=50)
    model.load(saver, save_path)
    trainer = Trainer(env, model, total_step, save_path, log_path, save_step, log_step)
    coord = tf.train.Coordinator()
    threads = []

    def trainer_fn():
        trainer.run(sess, saver, coord)

    thread = threading.Thread(target=trainer_fn)
    thread.start()
    threads.append(thread)
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
    coord.request_stop()
    coord.join(threads)
    save_flag = input('save final model? Y/N: ')
    if save_flag.lower().startswith('y'):
        print('saving model at step %d ...' % trainer.cur_step)
        model.save(saver, save_path + 'step', trainer.cur_step)

if __name__ == '__main__':
    single_gym_env()
