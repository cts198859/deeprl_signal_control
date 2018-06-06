"""
Particular class of small traffic network
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from envs.env import Phase, TrafficSimulator
from small_grid.data.build_file import gen_rou_file

sns.set_color_codes()

SMALL_GRID_NEIGHBOR_MAP = {'nt1': ['npc', 'nt2', 'nt6'],
                           'nt2': ['nt1', 'nt3'],
                           'nt3': ['npc', 'nt2', 'nt4'],
                           'nt4': ['nt3', 'nt5'],
                           'nt5': ['npc', 'nt4', 'nt6'],
                           'nt6': ['nt1', 'nt5']}

STATE_MEAN_MASKS = {'in_car': False, 'in_speed': True, 'out_car': False}
STATE_NAMES = ['in_car', 'in_speed', 'out_car']
# map from ild order (alphabeta) to signal order (clockwise from north)
STATE_PHASE_MAP = {'nt1': [0, 1, 2], 'nt2': [1, 0], 'nt3': [1, 0],
                   'nt4': [1, 0], 'nt5': [1, 0], 'nt6': [1, 0]}


class SmallGridPhase(Phase):
    def __init__(self):
        two_phase = []
        phase = {'green': 'GGrr', 'yellow': 'yyrr'}
        two_phase.append(phase)
        phase = {'green': 'rrGG', 'yellow': 'rryy'}
        two_phase.append(phase)
        three_phase = []
        phase = {'green': 'GGGrrrrrr', 'yellow': 'yyyrrrrrr'}
        three_phase.append(phase)
        phase = {'green': 'rrrGGGrrr', 'yellow': 'rrryyyrrr'}
        three_phase.append(phase)
        phase = {'green': 'rrrrrrGGG', 'yellow': 'rrrrrryyy'}
        three_phase.append(phase)
        self.phases = {2: two_phase, 3: three_phase}


class FixedController:
    def __init__(self, num_node_3phase=1, num_node_2phase=5, switch_step=2):
        self.name = 'naive'
        self.phase_3 = 0
        self.phase_2 = 0
        self.num_3 = num_node_3phase
        self.num_2 = num_node_2phase
        self.switch_step = switch_step
        self.step_3 = switch_step
        self.step_2 = switch_step

    def forward(self, ob=None, done=False, output_type=''):
        if not self.step_3:
            self.phase_3 = (self.phase_3 + 1) % 3
            self.step_3 = self.switch_step - 1
        else:
            self.step_3 -= 1
        if not self.step_2:
            self.phase_2 = (self.phase_2 + 1) % 2
            self.step_2 = self.switch_step - 1
        else:
            self.step_2 -= 1
        return np.array([self.phase_3] * self.num_3 + [self.phase_2] * self.num_2)


class SmallGridController:
    def __init__(self, nodes):
        self.name = 'naive'
        self.nodes = nodes

    def forward(self, obs):
        actions = []
        for ob, node in zip(obs, self.nodes):
            actions.append(self.greedy(ob, node))
        return actions

    def greedy(self, ob, node):
        # hard code the mapping from state to number of cars
        phases = STATE_PHASE_MAP[node]
        in_cars = ob[:len(phases)]
        return phases[np.argmax(in_cars)]


class SmallGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.num_car_hourly = config.getint('num_extra_car_per_hour')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_cross_action_num(self, node):
        return len(self.nodes[node].ild_in)

    def _init_map(self):
        self.neighbor_map = SMALL_GRID_NEIGHBOR_MAP
        self.phase_map = SmallGridPhase()
        self.state_names = STATE_NAMES
        self.state_mean_masks = STATE_MEAN_MASKS

    def _init_sim_config(self):
        return gen_rou_file(seed=self.seed,
                            thread=self.sim_thread,
                            path=self.data_path,
                            num_car_hourly=self.num_car_hourly)

    def plot_stat(self, rewards):
        data_set = {}
        data_set['car_num'] = np.array(self.car_num_stat)
        data_set['car_speed'] = np.array(self.car_speed_stat)
        data_set['reward'] = rewards
        for name, data in data_set.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = SmallGridEnv(config['ENV_CONFIG'], 2, base_dir, is_record=False, record_stat=True)
    ob = env.reset()
    controller = SmallGridController(env.control_nodes)
    rewards = []
    while True:
        next_ob, reward, done, _ = env.step(controller.forward(ob))
        rewards += list(reward)
        if done:
            break
        ob = next_ob
    env.plot_stat(np.array(rewards))
    env.terminate()
    time.sleep(2)
    # env.collect_tripinfo()
    # env.output_data()
