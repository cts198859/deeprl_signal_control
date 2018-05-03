"""
Traffic network simulator w/ defined sumo files
@author: Tianshu Chu
"""
import logging
import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import traci

DEFAULT_PORT = 8000


class Phase:
    def __init__(self):
        self.phases = {}

    def get_phase(self, phase_num, phase_type, action):
        # phase_type is either green or yellow
        return self.phases[phase_num][int(action)][phase_type]


class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control
        self.edge_in = []  # for reward
        self.footprint = []
        self.ild_in = []  # for state
        self.ild_out = []  # for state
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0
        self.num_footprint = 0
        self.state = []
        self.speed_mask = [] # speed is avg


class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        scenario = config.get('scenario')
        self.name = scenario
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        self.output_path = output_path
        self.scenario = scenario
        self.coop_level = config.get('coop_level')
        self.is_record = is_record
        self.record_stats = record_stats
        self.cur_episode = 0
        self.norm_car_num = config.getfloat('norm_car_num')
        self.norm_car_speed = config.getfloat('norm_car_speed')
        self.clip_car_num = config.getfloat('clip_car_num')
        self.clip_car_speed = config.getfloat('clip_car_speed')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        self.test_num = len(test_seeds)
        self.test_seeds = [int(s) for s in test_seeds]
        self._init_map()
        self._init_sim()
        self._init_nodes()
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
        if self.record_stats:
            self.car_num_stat = []
            self.car_speed_stat = []

    def _get_cross_phase(self, action, node, phase_type):
        phase_num = self.nodes[node].phase_num
        prev_action = self.nodes[node].prev_action
        if phase_type == 'green':
            return self.phase_map.get_phase(phase_num, 'green', action)
        self.nodes[node].prev_action = action
        if (prev_action >= 0) and (action != prev_action):
            yellow_phase = self.phase_map.get_phase(phase_num, 'yellow', prev_action)
            return yellow_phase
        green_phase = self.phase_map.get_phase(phase_num, 'green', action)
        return green_phase

    def _get_cross_action_num(self, node):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_cross_state_ilds(self, node, state_name):
        if state_name.split('_')[0] == 'in':
            return self.nodes[node].ild_in
        else:
            return self.nodes[node].ild_out

    def _get_cross_state_num(self, node):
        state_num = 0
        state_mean_mask = []
        for name in self.state_names:
            num = len(self._get_cross_state_ilds(node, name))
            state_mean_mask += [self.state_mean_masks[name]] * num
            state_num += num
        return state_num, np.array(state_mean_mask)

    def _get_state(self):
        state = []
        # cacluate mean aggrated state if necessary
        for node in self.all_nodes:
            cur_state = self.nodes[node].state
            is_speed = self.nodes[node].speed_mask
            speed_ind = np.where(is_speed)[0]
            carnum_ind = np.where(~is_speed)[0]
            speed_state = cur_state[speed_ind]
            speed_state /= float(self.control_interval_sec)
            cur_state[speed_ind] = self._norm_clip_state(speed_state,
                                                         self.norm_car_speed,
                                                         clip=self.clip_car_speed)
            carnum_state = cur_state[carnum_ind]
            cur_state[carnum_ind] = self._norm_clip_state(carnum_state,
                                                          self.norm_car_num,
                                                          clip=self.clip_car_num)
            self.nodes[node].state = cur_state
            if self.record_stats:
                self.car_num_stat += list(cur_state[carnum_ind])
                self.car_speed_stat += list(cur_state[speed_ind])

        # get the state vectors
        for node in self.control_nodes:
            if (self.coop_level == 'global') or (self.coop_level == 'local'):
                state.append(self.nodes[node].state)
            elif self.coop_level == 'neighbor':
                cur_state = [self.nodes[node].state]
                # TODO: add neighbor's footprint
                for nnode in self.nodes[node].neighbor:
                    cur_state.append(self.nodes[nnode].state)
                state.append(np.concatenate(cur_state))

        if self.coop_level == 'global':
            state = np.concatenate(state)

        # clean up the state measurements
        for node in self.all_nodes:
            self.nodes[node].state = np.zeros(self.nodes[node].num_state)
        return state

    def _init_nodes(self):
        nodes = {}
        for node in self.sim.trafficlight.getIDList():
            nodes[node] = Node(node,
                               neighbor=self.neighbor_map[node],
                               control=True)
        for loop in self.sim.inductionloop.getIDList():
            if loop.startswith('ild_in'):
                # the in of road ij is the out of cross i
                node = loop.split(':')[-1].split(',')[0]
                if node not in nodes:
                    nodes[node] = Node(node)
                nodes[node].ild_out.append(loop)
            else:
                # the out of road ji is the in of cross i
                node = loop.split(':')[-1].split(',')[1]
                if node not in nodes:
                    nodes[node] = Node(node)
                nodes[node].ild_in.append(loop)
                edge = 'e:' + loop.split(':')[-1]
                nodes[node].edge_in.append(edge)
        self.nodes = nodes
        self.all_nodes = sorted(list(nodes.keys()))
        self.control_nodes = [i for i in self.all_nodes if self.nodes[i].control]
        self.ilds = self.sim.inductionloop.getIDList()
        s = 'Env: init node information:\n'
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\tneigbor: %r\n' % node.neighbor
            s += '\tild_in: %r\n' % node.ild_in
            s += '\tild_out: %r\n' % node.ild_out
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        # for local and neighbor coop level
        self.n_a_ls = []
        for node in self.control_nodes:
            n_a = self._get_cross_action_num(node)
            self.nodes[node].phase_num = n_a
            self.n_a_ls.append(n_a)
        # for global coop level
        self.n_a = np.prod(np.array(self.n_a_ls))

    def _init_map(self):
        # needs to be overwriteen
        self.neighbor_map = None
        self.phase_map = None
        self.state_names = None
        self.state_mean_masks = None
        raise NotImplementedError()

    def _init_sim(self):
        sumocfg_file = self._init_sim_config()
        command = [checkBinary('sumo'), '-c', sumocfg_file]
        command += ['--seed', str(self.seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '-1']
        command += ['--no-warnings', 'True']
        subprocess.Popen(command)
        self.sim = traci.connect(port=self.port)

    def _init_sim_config(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _init_state_space(self):
        self._reset_state()
        self.n_s_ls = []
        for node in self.control_nodes:
            num_state = self.nodes[node].num_state
            if self.coop_level == 'neighbor':
                for nnode in self.nodes[node].neighbor:
                    num_state += self.nodes[nnode].num_footprint
            self.n_s_ls.append(num_state)
        self.n_s = np.sum(np.array(self.n_s_ls))

    def _measure_reward_step(self):
        rewards = []
        for node in self.control_nodes:
            reward = 0
            if self.obj == 'max_flow':
                for ild in self.nodes[node].ild_out:
                    reward += self.sim.inductionloop.getLastStepVehicleNumber(ild)
            elif self.obj == 'min_stop':
                for edge in self.nodes[node].edge_in:
                    reward -= self.sim.edge.getLastStepHaltingNumber(edge)
            elif self.obj == 'min_wait':
                for edge in self.nodes[node].edge_in:
                    reward -= self.sim.edge.getWaitingTime(edge)
            rewards.append(reward)
        return np.array(rewards)

    def _measure_state_step(self):
        for node in self.all_nodes:
            state = []
            for name in self.state_names:
                state_type = name.split('_')[1]
                for ild in self._get_cross_state_ilds(node, name):
                    if state_type == 'car':
                        state.append(self.sim.inductionloop.getLastStepVehicleNumber(ild))
                    elif state_type == 'speed':
                        state.append(self.sim.inductionloop.getLastStepMeanSpeed(ild))
            self.nodes[node].state += np.array(state)

    def _measure_traffic_step(self):
        cars = self.sim.vehicle.getIDList()
        num_tot_car = len(cars)
        speeds = np.array([self.sim.vehicle.getSpeed(car) for car in cars])
        # car is stopped if its speed < 0.5m/s
        num_stop_car = np.sum(speeds < 0.5)
        num_in_car = self.sim.simulation.getDepartedNumber()
        num_out_car = self.sim.simulation.getArrivedNumber()
        avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars])
        avg_speed = np.mean(speeds)
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'number_stopped_car': num_stop_car,
                       'average_waiting_time': avg_waiting_time,
                       'average_speed': avg_speed}
        self.traffic_data.append(cur_traffic)

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = x / norm
        return x if clip < 0 else np.clip(x, 0, clip)

    def _reset_state(self):
        for node in self.all_nodes:
            # prev action for yellow phase before each switch
            if self.nodes[node].control:
                # TODO: add num_footprint
                self.nodes[node].prev_action = -1
            num_state, speed_mask = self._get_cross_state_num(node)
            self.nodes[node].state = np.zeros(num_state)
            self.nodes[node].num_state = num_state
            self.nodes[node].speed_mask = speed_mask

    def _set_phase(self, action, phase_type, phase_duration):
        for node, a in zip(self.control_nodes, list(action)):
            phase = self._get_cross_phase(a, node, phase_type)
            self.sim.trafficlight.setRedYellowGreenState(node, phase)
            self.sim.trafficlight.setPhaseDuration(node, phase_duration)

    def _simulate(self, num_step):
        reward = np.zeros(len(self.control_nodes))
        for _ in range(num_step):
            self.sim.simulationStep()
            self._measure_state_step()
            reward += self._measure_reward_step()
            self.cur_sec += 1
            if self.is_record:
                self._measure_traffic_step()
        return reward

    def _transfer_action(self, action):
        '''Transfer global action to a list of local actions'''
        phase_nums = []
        for node in self.control_nodes:
            phase_nums.append(self.nodes[node].phase_num)
        action_ls = []
        for i in range(len(phase_nums) - 1):
            action, cur_action = divmod(action, phase_nums[i])
            action_ls.append(cur_action)
        action_ls.append(action)
        return action_ls

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + 'control.csv')
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + 'traffic.csv')

    def reset(self, test_ind=0):
        self.terminate()
        self._reset_state()
        if not self.train_mode:
            self.seed = self.test_seeds[test_ind]
        self._init_sim()
        # next environment random condition should be different
        self.seed += 10
        self.cur_sec = 0
        self.cur_episode += 1
        return self._get_state()

    def terminate(self):
        self.sim.close()

    def step(self, action):
        if self.coop_level == 'global':
            action = self._transfer_action(action)
        reward = np.zeros(len(self.control_nodes))
        self._set_phase(action, 'yellow', self.yellow_interval_sec)
        reward += self._simulate(self.yellow_interval_sec)
        rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec
        self._set_phase(action, 'green', rest_interval_sec)
        reward += self._simulate(rest_interval_sec)
        state = self._get_state()
        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True

        if self.is_record:
            action_str = ','.join([str(int(a)) for a in action])
            cur_control = {'episode': self.cur_episode,
                           'step': self.cur_sec / self.control_interval_sec,
                           'action': action_str,
                           'reward': reward}
            self.control_data.append(cur_control)
        if self.coop_level == 'global':
            reward = np.sum(reward)
        # TODO: neighbor uses spatially discounted reward 
        return state, reward, done
