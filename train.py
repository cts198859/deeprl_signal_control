import numpy as np
import tensorflow as tf


class Trainer:
    def __init__(self, env, model, total_step, save_path, log_path, save_step, log_step):
        self.cur_step = 0
        self.total_step = total_step
        self.cur_save_step = 0
        self.log_step = log_step
        self.save_path = save_path
        self.save_step = save_step
        self.env = env
        self.model = model
        self.n_step = self.model.n_step
        self.total_reward = tf.placeholder(tf.float32, [])
        self.actions = tf.placeholder(tf.int32, [None])
        summaries = []
        summaries.append(tf.summary.scalar('explore/total_reward', self.total_reward))
        summaries.append(tf.summary.histogram('explore/action', self.actions))
        self.summaries = tf.summary.merge(summaries)
        self.summary_writer = tf.summary.FileWriter(log_path)
        tf.logging.set_verbosity(tf.logging.INFO)

    def _add_summary(self, sess, cum_reward, cum_actions):
        # if len(cum_actions) > 50:
        #     cum_actions = cum_actions[:50]
        # else:
        #     cum_actions = cum_actions + [-1] * (50 - len(cum_actions))
        summ = sess.run(self.summaries, {self.total_reward:cum_reward, self.actions:cum_actions})
        self.summary_writer.add_summary(summ, global_step=self.cur_step)

    def explore(self, sess, prev_ob, prev_done, cum_reward, cum_actions):
        ob = prev_ob
        done = prev_done
        for _ in range(self.n_step):
            policy, value = self.model.forward(ob, done)
            action = np.random.choice(np.arange(len(policy)), p=policy)
            next_ob, reward, done, _ = self.env.step(action)
            cum_actions.append(action)
            cum_reward += reward
            self.cur_step += 1
            self.model.add_transition(ob, action, reward, value, done)
            # logging
            if self.cur_step % self.log_step == 0:
                tf.logging.info('cum step %d, temp step %d, ob: %s, a: %d, pi: %s, v: %.2f, r: %.2f, done: %r' % 
                                (self.cur_step - 1, len(cum_actions) - 1, str(ob), action, str(policy), value, reward, done))
            # termination
            if done:
                ob = self.env.reset()
                self._add_summary(sess, cum_reward, cum_actions)
                cum_reward = 0
                cum_actions = []
            else:
                ob = next_ob
        if done:
            R = 0
        else:
            R = self.model.forward(ob, False, 'v')
        return ob, done, R, cum_reward, cum_actions

    def run(self, sess, saver, coord):
        ob = self.env.reset()
        done = False
        cum_reward = 0
        cum_actions = []
        while not coord.should_stop():
            ob, done, R, cum_reward, cum_actions = self.explore(sess, ob, done, cum_reward, cum_actions)
            summ = self.model.backward(R)
            self.summary_writer.add_summary(summ, global_step=self.cur_step)
            self.summary_writer.flush()
            # save model
            if (self.cur_step - self.cur_save_step) >= self.save_step:
                print('saving model at step %d ...' % self.cur_step)
                self.model.save(saver, self.save_path + 'step', self.cur_step)
                self.cur_save_step = self.cur_step
            if self.cur_step >= self.total_step:
                coord.request_stop()


class AsyncTrainer(Trainer):
    def __init__(self, i_thread, env, model, global_model=None):
        pass
