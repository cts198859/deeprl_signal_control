import configparser
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


class SmallGridController:
    def __init__(self, num_node_3phase=1, num_node_2phase=5, switch_step=2):
        self.phase_3 = 0
        self.phase_2 = 0
        self.num_3 = num_node_3phase
        self.num_2 = num_node_2phase
        self.switch_step = switch_step
        self.step_3 = switch_step
        self.step_2 = switch_step

    def act(self, state=None):
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

    def explore(self, state=None):
        return self.act()


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
            fig.savefig(self.output_path + name + '.png')

def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./config/config_local.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = SmallGridEnv(config['ENV_CONFIG'], 0, base_dir, is_record=True, record_stat=True)
    env.reset()
    controller = SmallGridController()
    rewards = []
    while True:
        _, reward, done, _ = env.step(controller.act())
        rewards.append(np.mean(reward))
        if done:
            break
    env.plot_stat(np.array(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
