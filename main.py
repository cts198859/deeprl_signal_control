import argparse
import configparser
import logging
import tensorflow as tf
import threading
from envs.small_grid_env import SmallGridEnv
from agents.models import A2C, MultiA2C
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag,
                   plot_evaluation, plot_train)

AGENTS = {'global': 'A2C', 'local': 'IA2C', 'neighbor': 'MA2C'}

def parse_args():
    default_base_dir = '/rl_test/small_grid/global'
    default_config_dir = '/deeprl_signal_control/config/config_global.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--test-mode', type=str, required=False,
                    default='no_test',
                    help="test mode during training",
                    choices=['no_test', 'in_train_test', 'after_train_test', 'all_test'])
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--agents', type=str, required=False,
                    default='', help="agent folder names for evaluation, split by ,")
    sp.add_argument('--evaluate-metrics', type=str, required=False,
                    default='num_arrival_car',
                    help="cumulative evaluation metrics over planning horizon",
                    choices=['num_arrival_car', 'trip_waiting_time', 'trip_total_time'])
    sp.add_argument('--evaluate-seeds', type=str, required=False,
                    default='', help="random seeds for evaluation, split by ,")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    in_test, post_test = init_test_flag(args.test_mode)

    # init env
    env = SmallGridEnv(config['ENV_CONFIG'], port=0)
    logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    coord = tf.train.Coordinator()

    if env.coop_level == 'global':
        model = A2C(env.n_s, env.n_a, total_step,
                    config['MODEL_CONFIG'], seed=seed)
    else:
        model = MultiA2C(env.n_s_ls, env.n_a_ls, total_step,
                         config['MODEL_CONFIG'], seed=seed)

    threads = []
    summary_writer = tf.summary.FileWriter(dirs['log'])
    trainer = Trainer(env, model, global_counter, summary_writer)
    if in_test or post_test:
        # assign a different port for test env
        test_env = SmallGridEnv(config['ENV_CONFIG'], port=1)
        tester = Tester(test_env, model, global_counter, summary_writer)

    def train_fn():
        trainer.run(coord)

    thread = threading.Thread(target=train_fn)
    thread.start()
    threads.append(thread)
    if in_test:
        def test_fn():
            tester.run_online(coord)
        thread = threading.Thread(target=test_fn)
        thread.start()
        threads.append(thread)
    coord.join(threads)

    # post-training test
    if post_test:
        tester.run_offline(dirs['data'])

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'], final_step)
    env.terminate()
    if in_test or post_test:
        test_env.terminate()


def evaluate(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log', 'eva_plot'])
    init_log(dirs['eva_log'])
    agents = args.agents.split(',')
    seeds = args.evaluate_seeds.split(',')
    seeds = [int(s) for s in seeds]
    train_data_dirs = []
    eva_data_dirs = []
    train_labels = []
    eva_labels = []
    for agent in agents:
        cur_dir = base_dir + '/' + agent
        if not check_dir(cur_dir):
            logging.error('%s does not exist!' % agent)
            continue
        config_dir = find_file(cur_dir + '/data')
        if not config_dir:
            continue
        config = configparser.ConfigParser()
        config.read(config_dir)

        # collect training data
        train_data_dir = find_file(cur_dir + '/data', suffix='train_reward.csv')
        if train_data_dir:
            train_data_dirs.append(train_data_dir)
            train_labels.append(AGENTS[agent])
        # init env
        env = SmallGridEnv(config['ENV_CONFIG'], port=0)
        logging.info('Evaluation: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                     (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))
        # enforce the same evaluation seeds across agents
        env.init_test_seeds(seeds)

        # init centralized or multi agent
        if env.coop_level == 'global':
            model = A2C(env.n_s, env.n_a, 0, config['MODEL_CONFIG'])
        else:
            model = MultiA2C(env.n_s_ls, env.n_a_ls, 0, config['MODEL_CONFIG'])
        if not model.load(cur_dir + '/model'):
            continue

        # collect evaluation data
        evaluator = Evaluator(env, model, args.evaluate_metrics)
        eva_data_dirs.append(evaluator.run())
        eva_labels.append(AGENTS[agent])

    if len(train_data_dirs):
        plot_train(train_data_dirs, train_labels)
    if len(eva_data_dirs):
        plot_evaluation(eva_data_dirs, eva_labels)


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
