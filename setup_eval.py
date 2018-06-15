import argparse
import os
import subprocess


def parse_args():
    default_train_dir = 'tests_may22'
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
    for scenario in ['small_grid', 'large_grid']:
        if not os.path.exists(eval_dir + '/' + scenario):
            os.mkdir(eval_dir + '/' + scenario)
        # create folder for train reward
        train_data_dir = '/'.join([eval_dir, scenario, 'train_data'])
        if not os.path.exists(train_data_dir):
            os.mkdir(train_data_dir)
        for coop in ['neighbor', 'global', 'local']:
            case = scenario + '_' + coop
            if case not in folders:
                continue
            cur_folder = train_dir + '/' + case
            # copy config file
            cmd = 'cp %s %s' % (cur_folder + '/data/*.ini', cur_folder + '/model/')
            subprocess.check_call(cmd, shell=True)
            # copy model
            cmd = 'cp -r %s %s' % (cur_folder + '/model', '/'.join([eval_dir, scenario, coop]))
            subprocess.check_call(cmd, shell=True)
            # copy train reward
            tar_train_file = train_data_dir + '/' + coop + '_train_reward.csv'
            cmd = 'cp %s %s' % (cur_folder + '/log/train_reward.csv', tar_train_file)
            subprocess.check_call(cmd, shell=True)
        new_folder = '/'.join([eval_dir, scenario, 'naive'])
        old_folder = '/'.join([eval_dir, scenario, 'local'])
        os.mkdir(new_folder)
        cmd = 'cp %s %s' % (old_folder + '/*.ini', new_folder + '/')
        subprocess.check_call(cmd, shell=True)

if __name__ == '__main__':
    main()
