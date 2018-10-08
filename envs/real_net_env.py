"""
Particular class of real traffic network
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
# from real_net.data.build_file import gen_rou_file

sns.set_color_codes()

STATE_NAMES = ['wave', 'wait']


class RealNetPhase(PhaseMap):
    def __init__(self):
        phases = ['GGgrrrGGgrrr', 'rrrGrGrrrGrG', 'rrrGGrrrrGGr',
                  'rrrGGGrrrrrr', 'rrrrrrrrrGGG']
        self.phases = {5: PhaseSet(phases)}


class RealNetEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.peak_flow1 = config.getint('peak_flow1')
        self.peak_flow2 = config.getint('peak_flow2')
        self.init_density = config.getfloat('init_density')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node):
        return 5

    def _init_neighbor_map(self):
        return dict()

    def _init_map(self):
        self.neighbor_map = self._init_neighbor_map()
        self.phase_map = RealNetPhase()
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        # comment out to call build_file.py
        return self.data_path + 'most.sumocfg'
        # return gen_rou_file(self.data_path,
        #                     self.peak_flow1,
        #                     self.peak_flow2,
        #                     self.init_density,
        #                     seed=seed,
        #                     thread=self.sim_thread)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test_real.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = RealNetEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    env.train_mode = False
