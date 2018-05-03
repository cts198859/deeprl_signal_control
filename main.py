import argparse
import configparser
import logging
import tensorflow as tf
import threading
from envs.small_grid_env import SmallGridEnv
from agents.models import A2C
from utils import (Counter, Trainer, Tester,
                   init_dir, init_log, init_test_flag
                   )


def parse_args():
    default_base_dir = '/Users/tchu/Documents/rl_test/small_grid0'
    default_config_dir = '/Users/tchu/Dropbox/signal-ITS/code/config/config.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--test-mode', type=str, reauired=False,
                        default='no_test',
                        help="select from: no_test, in_train_test, after_train_test, all_test")
    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    in_test, post_test = init_test_flag(args.test_mode)

    # init env
    env = SmallGridEnv(config['ENV_CONFIG'], port=0)
    if in_test:
        test_env = SmallGridEnv(config['ENV_CONFIG'], port=1)
    logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))

    # init step counter
    total_step_min = int(config.getfloat('TRAIN_CONFIG', 'total_step_min'))
    total_step_max = int(config.getfloat('TRAIN_CONFIG', 'total_step_max'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    delta_reward = config.getfloat('TRAIN_CONFIG', 'delta_reward')
    global_counter = Counter(total_step_min, total_step_max, test_step,
                             log_step, delta_reward)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    coord = tf.train.Coordinator()

    if env.coop_level == 'global':
        model = A2C(env.n_s, env.n_a, total_step_min,
                    config['MODEL_CONFIG'], seed=seed)
    else:
        model = None

    threads = []
    summary_writer = tf.summary.FileWriter(dirs['log'])
    trainer = Trainer(env, model, global_counter, summary_writer)
    if in_test:
        tester = Tester(test_env, model, global_counter, summary_writer)

    def train_fn():
        trainer.run(coord)

    def test_fn():
        tester.run(coord)

    thread = threading.Thread(target=train_fn)
    thread.start()
    threads.append(thread)
    if in_test:
        thread = threading.Thread(target=test_fn)
        thread.start()
        threads.append(thread)
    coord.join(threads)

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'], final_step)


if __name__ == '__main__':
    train()
