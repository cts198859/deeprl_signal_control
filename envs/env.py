"""
Traffic network simulator w/ defined sumo files
@author: Tianshu Chu
"""
import logging
import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci
import xml.etree.cElementTree as ET

DEFAULT_PORT = 8000
SEC_IN_MS = 1000


class PhaseSet:
    def __init__(self, phases):
        self.phases = {}
        self.num_phase = len(phases)
        self.num_lane = len(phases[0])
        self.phases['green'] = phases
        self._init_phase_set()

    @staticmethod
    def _get_phase_red_lanes(phase):
        red_lanes = []
        for i, l in enumerate(phase):
            if l == 'r':
                red_lanes.append(i)
        return red_lanes

    @staticmethod
    def _get_yellow_phase(phase):
        yellow_phase = ''
        for l in phase:
            yl = 'y' if l in 'gG' else l
            yellow_phase += yl
        return yellow_phase

    def _init_phase_set(self):
        self.red_lanes = []
        yellow_phases = []
        for phase in self.phases['green']:
            self.red_lanes.append(self._get_phase_red_lanes(phase))
            yellow_phases.append(self._get_yellow_phase(phase))
        self.phases['yellow'] = yellow_phases


class PhaseMap:
    def __init__(self):
        self.phases = {}

    def get_phase(self, phase_id, phase_type, action):
        # phase_type is either green or yellow
        return self.phases[phase_id].phases[phase_type][int(action)]

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase

    def get_lane_num(self, phase_id):
        # the lane number is link number
        return self.phases[phase_id].num_lane

    def get_red_lanes(self, phase_id, action):
        # the lane number is link number
        return self.phases[phase_id].red_lanes[int(action)]


class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control # disabled
        self.edges_in = []  # for reward
        self.lanes_in = []
        self.ilds_in = [] # for state
        self.fingerprint = [] # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0
        self.num_fingerprint = 0
        self.state = [] # local state
        self.waits = []
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1


class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        self.name = config.get('scenario')
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]
        self._init_map()
        self.init_data(is_record, record_stats, output_path)
        self.init_test_seeds(test_seeds)
        self._init_sim()
        self._init_nodes()
        self.terminate()

    def _debug_traffic_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase = self.sim.trafficlight.getRedYellowGreenState(self.node_names[0])
            cur_traffic = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'node': node_name,
                           'action': node.prev_action,
                           'phase': phase}
            for i, ild in enumerate(node.ilds_in):
                cur_name = 'lane%d_' % i
                lane_name = 'e:' + ild.split(':')[1]
                cur_traffic[cur_name + 'queue'] = self.sim.lane.getLastStepHaltingNumber(lane_name)
                cur_traffic[cur_name + 'flow'] = self.sim.lane.getLastStepVehicleNumber(lane_name)
                cur_traffic[cur_name + 'wait'] = node.waits[i]
            self.traffic_data.append(cur_traffic)

    def _get_node_phase(self, action, node_name, phase_type):
        node = self.nodes[node_name]
        phase_num = node.n_a
        if phase_type == 'green':
            return self.phase_map.get_phase(phase_num, 'green', action)
        prev_action = node.prev_action
        node.prev_action = action
        if (prev_action >= 0) and (action != prev_action):
            yellow_phase = self.phase_map.get_phase(phase_num, 'yellow', prev_action)
            return yellow_phase
        green_phase = self.phase_map.get_phase(phase_num, 'green', action)
        return green_phase

    def _get_node_phase_id(self, node_name):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_node_state_num(self, node):
        assert len(node.lanes_in) == self.phase_map.get_lane_num(node.phase_id)
        # wait + wave states for each lane
        return len(self.state_names) * len(node.ilds_in)

    def _get_state(self):
        state = []
        # measure the most recent state
        self._measure_state_step()

        # get the appropriate state vectors
        for node in self.node_names:
            if self.agent in ['greedy', 'a2c']:
                state.append(self.nodes[node].state)
            else:
                cur_state = [self.nodes[node].state]
                # include states of neighbors
                for nnode_name in self.nodes[node].neighbor:
                    if self.agent != 'ma2c':
                        cur_state.append(self.nodes[nnode_name].state)
                    else:
                        # discount the neigboring states
                        cur_state.append(self.nodes[nnode_name].state * self.coop_gamma)
                # include fingerprints of neighbors
                if self.agent == 'ma2c':
                    for nnode_name in self.nodes[node].neighbor:
                        cur_state.append(self.nodes[nnode_name].fingerprint)
                state.append(np.concatenate(cur_state))

        if self.agent == 'a2c':
            state = np.concatenate(state)

        # # clean up the state and fingerprint measurements
        # for node in self.node_names:
        #     self.nodes[node].state = np.zeros(self.nodes[node].num_state)
        #     self.nodes[node].fingerprint = np.zeros(self.nodes[node].num_fingerprint)
        return state

    def _init_nodes(self):
        nodes = {}
        for node_name in self.sim.trafficlight.getIDList():
            nodes[node_name] = Node(node_name,
                                    neighbor=self.neighbor_map[node_name],
                                    control=True)
            # controlled lanes: l:j,i_k
            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)
            nodes[node_name].lanes_in = lanes_in
            # controlled edges: e:j,i
            # lane ilds: ild:j,i_k for road ji, lane k.
            edges_in = []
            ilds_in = []
            for lane_name in lanes_in:
                edge_name = 'e:' + lane_name.split(':')[-1].split('_')[0]
                if edge_name not in edges_in:
                    edges_in.append(edge_name)
                ild_name = 'ild:' + lane_name.split(':')[-1]
                if ild_name not in ilds_in:
                    ilds_in.append(ild_name)
            nodes[node_name].edges_in = edges_in
            nodes[node_name].ilds_in = ilds_in
        self.nodes = nodes
        self.node_names = sorted(list(nodes.keys()))
        s = 'Env: init node information:\n'
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\tneigbor: %r\n' % node.neighbor
            s += '\tilds_in: %r\n' % node.ilds_in
            s += '\tedges_in: %r\n' % node.edges_in
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        # for local and neighbor coop level
        self.n_a_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_id = self._get_node_phase_id(node_name)
            node.phase_id = phase_id
            node.n_a = self.phase_map.get_phase_num(phase_id)
            self.n_a_ls.append(node.n_a)
        # for global coop level
        self.n_a = np.prod(np.array(self.n_a_ls))

    def _init_map(self):
        # needs to be overwriteen
        self.neighbor_map = None
        self.phase_map = None
        self.state_names = None
        raise NotImplementedError()

    def _init_policy(self):
        policy = []
        for node_name in self.node_names:
            phase_num = self.nodes[node_name].n_a
            p = 1. / phase_num
            policy.append(np.array([p] * phase_num))
        return policy

    def _init_sim(self, gui=False):
        sumocfg_file = self._init_sim_config()
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--seed', str(self.seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        if self.name != 'large_grid':
            command += ['--time-to-teleport', '-1'] # disable teleport
        else:
            command += ['--time-to-teleport', '180']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        if self.is_record:
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        subprocess.Popen(command)
        # wait 5s to establish the traci server
        time.sleep(5)
        self.sim = traci.connect(port=self.port)

    def _init_sim_config(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _init_state_space(self):
        self._reset_state()
        self.n_s_ls = []
        self.n_f_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            num_state = node.num_state
            num_fingerprint = 0
            for nnode_name in node.neighbor:
                if self.agent not in ['a2c', 'greedy']:
                    # all marl agents have neighborhood communication
                    num_state += self.nodes[nnode_name].num_state
                if self.agent == 'ma2c':
                    # only ma2c uses neighbor's policy
                    num_fingerprint += self.nodes[nnode_name].num_fingerprint
            self.n_s_ls.append(num_state + num_fingerprint)
            self.n_f_ls.append(num_fingerprint)
        self.n_s = np.sum(np.array(self.n_s_ls))

    def _measure_reward_step(self):
        rewards = []
        for node_name in self.node_names:
            queues = []
            for ild in self.nodes[node_name].ilds_in:
                lane_name = 'e:' + ild.split(':')[1]
                queues.append(self.sim.lane.getLastStepHaltingNumber(lane_name))
            queues = np.array(queues)
            if self.obj in ['queue', 'hybrid']:
                queue = np.sum(queues)
            if self.obj in ['wait', 'hybrid']:
                wait = np.sum(self.nodes[node_name].waits * (queues > 0))
            if self.obj == 'queue':
                reward = - queue
            elif self.obj == 'wait':
                reward = - wait
            else:
                reward = - queue - self.coef_wait * wait
            rewards.append(reward)
        return np.array(rewards)

    def _measure_state_step(self):
        for node_name in self.node_names:
            state = []
            for state_name in self.state_names:
                if state_name == 'wave':
                    cur_state = []
                    for ild in self.nodes[node_name].ilds_in:
                        cur_state.append(self.sim.lanearea.getLastStepVehicleNumber(ild))
                    cur_state = np.array(cur_state)
                elif state_name == 'wait':
                    cur_state = self.nodes[node_name].waits
                if self.record_stats:
                    self.state_stat[state_name] += list(cur_state)
                # normalization
                norm_cur_state = self._norm_clip_state(cur_state,
                                                       self.norms[state_name],
                                                       self.clips[state_name])
                state.append(norm_cur_state)
            self.nodes[node_name].state = np.concatenate(state)

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
        # all trip-related measurements are not supported by traci,
        # need to read from outputfile afterwards
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
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # prev action for yellow phase before each switch
            node.prev_action = 0
            # fingerprint is previous policy[:-1]
            node.num_fingerprint = node.n_a - 1
            node.num_state = self._get_node_state_num(node)
            node.state = np.zeros(node.num_state)
            node.waits = np.zeros(len(node.ilds_in))

    def _set_phase(self, action, phase_type, phase_duration):
        for node_name, a in zip(self.node_names, list(action)):
            phase = self._get_node_phase(a, node_name, phase_type)
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
            self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)

    def _simulate(self, num_step):
        # reward = np.zeros(len(self.control_node_names))
        for _ in range(num_step):
            self.sim.simulationStep()
            # self._measure_state_step()
            # reward += self._measure_reward_step()
            self.cur_sec += 1
            if self.is_record:
                self._debug_traffic_step()
                # self._measure_traffic_step()
        # return reward

    def _transfer_action(self, action):
        '''Transfer global action to a list of local actions'''
        phase_nums = []
        for node in self.control_node_names:
            phase_nums.append(self.nodes[node].phase_num)
        action_ls = []
        for i in range(len(phase_nums) - 1):
            action, cur_action = divmod(action, phase_nums[i])
            action_ls.append(cur_action)
        action_ls.append(action)
        return action_ls

    def _update_waits(self, action):
        for node_name, a in zip(self.node_names, action):
            red_lanes = set()
            node = self.nodes[node_name]
            for i in self.phase_map.get_red_lanes(node.phase_id, a):
                red_lanes.add(node.lanes_in[i])
            for i in range(len(node.waits)):
                lane = 'e:' + node.ilds_in[i].split(':')[1]
                if lane in red_lanes:
                    node.waits[i] += self.control_interval_sec
                else:
                    node.waits[i] = 0

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))
        tree = ET.ElementTree(file=trip_file)
        for child in tree.getroot():
            cur_trip = child.attrib
            cur_dict = {}
            cur_dict['episode'] = self.cur_episode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitSteps']
            cur_dict['wait_sec'] = cur_trip['timeLoss']
            self.trip_data.append(cur_dict)
        # delete the current xml
        cmd = 'rm ' + trip_file
        subprocess.check_call(cmd, shell=True)

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.record_stats = record_stats
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []
        if self.record_stats:
            self.state_stat = {}
            for state_name in self.state_names:
                self.state_stat[state_name] = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.agent)))

    def reset(self, test_ind=0):
        # have to terminate previous sim before calling reset
        self._reset_state()
        if not self.train_mode:
            self.seed = self.test_seeds[test_ind]
        # self._init_sim(gui=True)
        self._init_sim()
        # next environment random condition should be different
        self.seed += 10
        self.cur_sec = 0
        self.cur_episode += 1
        # initialize fingerprint
        if self.agent == 'ma2c':
            self.update_fingerprint(self._init_policy())
        return self._get_state()

    def terminate(self):
        self.sim.close()

    def step(self, action):
        if self.agent == 'a2c':
            action = self._transfer_action(action)
        self._update_waits(action)
        self._set_phase(action, 'yellow', self.yellow_interval_sec)
        self._simulate(self.yellow_interval_sec)
        rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec
        self._set_phase(action, 'green', rest_interval_sec)
        self._simulate(rest_interval_sec)
        state = self._get_state()
        reward = self._measure_reward_step()
        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True

        if self.is_record:
            if self.name == 'large_grid':
                node_name = 'nt13'
            node_ind = self.node_names.index(node_name)
            action_r = int(action[node_ind])
            reward_r = reward[node_ind]
            state_r = ','.join(['%.2f' % s for s in state[node_ind]])
            cur_control = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'step': self.cur_sec / self.control_interval_sec,
                           'state': state_r,
                           'action': action_r,
                           'reward': reward_r}
            self.control_data.append(cur_control)
        global_reward = np.sum(reward) # for fair comparison
        # use local rewards in test
        if not self.train_mode:
            return state, reward, done, global_reward
        if self.agent in ['a2c', 'greedy']:
            reward = global_reward
        elif self.agent != 'ma2c':
            # global reward is shared in independent rl
            new_reward = [global_reward] * len(reward)
            reward = np.array(new_reward)
        else:
            # discounted global reward
            new_reward = []
            for node, r in zip(self.node_names, reward):
                cur_reward = r
                for i, nnode in enumerate(self.node_names):
                    if nnode == node:
                        continue
                    if nnode in self.nodes[node].neighbor:
                        cur_reward += self.coop_gamma * reward[i]
                    elif self.name == 'small_grid':
                        # in small grid, agent is at most 2 steps away
                        cur_reward += (self.coop_gamma ** 2) * reward[i]
                    else:
                        # in large grid, a distance map is used
                        if nnode in self.distance_map[node]:
                            distance = self.distance_map[node][nnode]
                            cur_reward += (self.coop_gamma ** distance) * reward[i]
                        else:
                            cur_reward += (self.coop_gamma ** self.max_distance) * reward[i]
                new_reward.append(cur_reward)
            reward = np.array(new_reward)
        return state, reward, done, global_reward

    def update_fingerprint(self, policy):
        for node_name, pi in zip(self.node_names, policy):
            self.nodes[node_name].fingerprint = np.array(pi)[:-1]
