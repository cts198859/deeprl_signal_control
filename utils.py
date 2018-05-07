import itertools
import logging
import numpy as np
import tensorflow as tf
import time
import os


def init_dir(base_dir):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in ['log', 'data', 'model', 'plot']:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    # def update_test(self, reward):
    #     if self.prev_reward is not None:
    #         if abs(self.prev_reward - reward) <= self.delta_reward:
    #             self.stop = True
    #     self.prev_reward = reward

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.coop_level = self.env.coop_level
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        self._init_summary()

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('train_reward', self.reward)

    def _add_summary(self, reward, global_step):
        summ = self.sess.run(self.summary, {self.reward:reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob, prev_done, cum_reward):
        ob = prev_ob
        done = prev_done
        for _ in range(self.n_step):
            policy, value = self.model.forward(ob, done)
            # need to update fingerprint before calling step
            if self.coop_level == 'neighbor':
                self.env.update_fingerprint(policy)
            if self.coop_level == 'global':
                action = np.random.choice(np.arange(len(policy)), p=policy)
            else:
                action = []
                for pi in policy:
                    action.append(np.random.choice(np.arange(len(pi)), p=pi))
            next_ob, reward, done, global_reward = self.env.step(action)
            cum_reward += global_reward
            global_step = self.global_counter.next()
            self.cur_step += 1
            self.model.add_transition(ob, action, reward, value, done)
            # logging
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %.2f, pi: %s, v: %.2f, r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), action, str(policy), value, reward, done))
            # termination
            if done:
                ob = self.env.reset()
                self._add_summary(cum_reward / float(self.cur_step), global_step)
                cum_reward = 0
                self.cur_step = 0
            else:
                ob = next_ob
        if done:
            R = 0 if self.coop_level == 'global' else [0] * self.model.n_agent
        else:
            R = self.model.forward(ob, False, 'v')
        return ob, done, R, cum_reward

    def run(self, coord):
        ob = self.env.reset()
        done = False
        cum_reward = 0
        while not coord.should_stop():
            ob, done, R, cum_reward = self.explore(ob, done, cum_reward)
            global_step = self.global_counter.cur_step
            self.model.backward(R, self.summary_writer, global_step)
            self.summary_writer.flush()
            if (self.global_counter.should_stop()) and (not coord.should_stop()):
                coord.request_stop()
                logging.info('Training: stop condition reached!')
                return


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def perform(self, test_ind):
        ob = self.env.reset(test_ind=test_ind)
        rewards = []
        while True:
            policy = self.model.forward(ob, False, 'p')
            if self.coop_level == 'neighbor':
                self.env.update_fingerprint(policy)
            if self.coop_level == 'global':
                action = np.argmax(np.array(policy))
            else:
                action = []
                for pi in policy:
                    action.append(np.argmax(np.array(pi)))
            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        total_reward = np.mean(np.array(rewards))
        return total_reward

    def run(self, coord):
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                for test_ind in range(self.test_num):
                    rewards.append(self.perform(test_ind))
                avg_reward = np.mean(np.array(rewards))
                global_step = self.global_counter.cur_step
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        