import argparse
import os
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer

SIZE_GUIDANCE = {'scalars': 5000}


def parse_args():
    default_log_dir = '~/tests_jun06/large_grid_neighbor/log/'
    default_scalar_name = 'train_reward'
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, required=False,
                        default=default_log_dir, help="dir of tensorboard logs")
    parser.add_argument('--scalar-name', type=str, required=False,
                        default=default_scalar_name, help="scalar names to be extraced")
    return parser.parse_args()


def create_multiplexer(run_path, run_name):
    run_path_map = {run_name: run_path}
    multiplexer = event_multiplexer.EventMultiplexer(run_path_map=run_path_map,
                                                     tensor_size_guidance=SIZE_GUIDANCE)
    multiplexer.Reload()
    return multiplexer


def extract_scalar(multiplexer, run_name, tag):
    tensor_events = multiplexer.Tensors(run_name, tag)
    data = {'wall_time': [], 'step': [], 'value': []}
    for event in tensor_events:
        data['wall_time'].append(event.wall_time)
        data['step'].append(event.step)
        data['value'].append(tf.make_ndarray(event.tensor_proto).item())
    return pd.DataFrame(data)


def get_event_log(log_dir):
    for f in os.listdir(log_dir):
        if f.startswith('events.out.tfevents'):
            return (log_dir + f)
    return None


def main():
    args = parse_args()
    log_dir = args.log_dir
    tag = args.scalar_name
    run_name = 'tfevent'
    run_path = get_event_log(log_dir)
    if not run_path:
        exit(1)
    multiplexer = create_multiplexer(run_path, run_name)
    df = extract_scalar(multiplexer, run_name, tag)
    df.to_csv(log_dir + tag + '.csv')


if __name__ == '__main__':
    main()
