import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect


class Policy:
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name):
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_fc_net(self, h, n_fc, out_type):
        h = fc(h, out_type + '_fc', n_fc)
        if out_type == 'pi':
            pi = fc(h, out_type, self.n_a, act=tf.nn.softmax)
            return tf.squeeze(pi)
        else:
            v = fc(h, out_type, 1, act=lambda x: x)
            return tf.squeeze(v)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi)
        if 'v' in out_type:
            outs.append(self.v)
        return outs

    def _return_forward_outs(self, out_values):
        if len(out_values) == 1:
            return out_values[0]
        return out_values

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables(scope=self.name)
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.lr = tf.placeholder(tf.float32, [])
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha,
                                                   epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(list(zip(grads, wts)))
        # monitor training
        summaries = []
        summaries.append(tf.summary.scalar('loss/entropy_loss', entropy_loss))
        summaries.append(tf.summary.scalar('loss/policy_loss', policy_loss))
        summaries.append(tf.summary.scalar('loss/value_loss', value_loss))
        summaries.append(tf.summary.scalar('loss/total_loss', self.loss))
        summaries.append(tf.summary.scalar('train/lr', self.lr))
        summaries.append(tf.summary.scalar('train/entropy_beta', self.entropy_coef))
        summaries.append(tf.summary.scalar('train/gradnorm', self.grad_norm))
        self.summary = tf.summary.merge(summaries)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_step, n_fc=128, n_lstm=64, name=None):
        super().__init__(n_a, n_s, n_step, 'lstm', name)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [2, n_lstm * 2])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi_fw, pi_state = self._build_net(n_fc, 'forward', 'pi')
            self.v_fw, v_state = self._build_net(n_fc, 'forward', 'v')
            pi_state = tf.expand_dims(pi_state, 0)
            v_state = tf.expand_dims(v_state, 0)
            self.new_states = tf.concat([pi_state, v_state], 0)
        with tf.variable_scope(self.name, reuse=True):
            self.pi, _ = self._build_net(n_fc, 'backward', 'pi')
            self.v, _ = self._build_net(n_fc, 'backward', 'v')
        self._reset()

    def _build_net(self, n_fc, in_type, out_type):
        if in_type == 'forward':
            ob = self.ob_fw
            done = self.done_fw
        else:
            ob = self.ob_bw
            done = self.done_bw
        if out_type == 'pi':
            states = self.states[0]
        else:
            states = self.states[1]
        h, new_states = lstm(ob, done, states, out_type + '_lstm')
        out_val = self._build_fc_net(h, n_fc, out_type)
        return out_val, new_states

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)
        self.states_bw = np.zeros((2, self.n_lstm * 2), dtype=np.float32)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        # update state only when p is called
        if 'p' in out_type:
            outs.append(self.new_states)
        out_values = sess.run(outs, {self.ob_fw:np.array([ob]),
                                     self.done_fw:np.array([done]),
                                     self.states:self.states_fw})
        if 'p' in out_type:
            self.states_fw = out_values[-1]
            out_values = out_values[:-1]
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        summary, _ = sess.run([self.summary, self._train],
                              {self.ob_bw:obs,
                               self.done_bw:dones,
                               self.states:self.states_bw,
                               self.A:acts,
                               self.ADV:Advs,
                               self.R:Rs,
                               self.lr:cur_lr,
                               self.entropy_coef:cur_beta})
        self.states_bw = np.copy(self.states_fw)
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step=global_step)

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            if self.discrete:
                outs.append(self.pi_fw)
            else:
                outs += self.pi_fw
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs


class Cnn1DPolicy(Policy):
    def __init__(self, n_s, n_a, n_step, n_past=10, n_fc=128,
                 n_filter=64, m_filter=4, name=None):
        super().__init__(n_a, n_s, n_step, 'cnn', name)
        self.n_past = n_past
        self.n_filter = n_filter
        self.m_filter = m_filter
        self.obs = tf.placeholder(tf.float32, [None, n_past, n_s])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net(n_fc, 'pi')
            self.v = self._build_net(n_fc, 'v')
        self._reset()

    def _build_net(self, n_fc, out_type):
        h = conv(self.obs, out_type + '_conv1', self.n_filter, self.m_filter, conv_dim=1)
        n_conv_fc = np.prod([v.value for v in h.shape[1:]])
        h = tf.reshape(h, [-1, n_conv_fc])
        return self._build_fc_net(h, n_fc, out_type)

    def _reset(self):
        self.recent_obs_fw = np.zeros((self.n_past-1, self.n_s))
        self.recent_obs_bw = np.zeros((self.n_past-1, self.n_s))
        self.recent_dones_fw = np.zeros(self.n_past-1)
        self.recent_dones_bw = np.zeros(self.n_past-1)

    def _recent_ob(self, obs, dones, ob_type='forward'):
        # convert [n_step, n_s] to [n_step, n_past, n_s]
        num_obs = len(obs)
        if ob_type == 'forward':
            recent_obs = np.copy(self.recent_obs_fw)
            recent_dones = np.copy(self.recent_dones_fw)
        else:
            recent_obs = np.copy(self.recent_obs_bw)
            recent_dones = np.copy(self.recent_dones_bw)
        comb_obs = np.vstack([recent_obs, obs])
        comb_dones = np.concatenate([recent_dones, dones])
        new_obs = []
        inds = list(np.nonzero(comb_dones)[0])
        for i in range(num_obs):
            cur_obs = np.copy(comb_obs[i:(i + self.n_past)])
            # print(cur_obs)
            if len(inds):
                k = bisect.bisect_left(inds, (i + self.n_past)) - 1
                if (k >= 0) and (inds[k] > i):
                    cur_obs[:(int(inds[k]) - i)] *= 0
            new_obs.append(cur_obs)
        recent_obs = comb_obs[(1-self.n_past):]
        recent_dones = comb_dones[(1-self.n_past):]
        if ob_type == 'forward':
            self.recent_obs_fw = recent_obs
            self.recent_dones_fw = recent_dones
        else:
            self.recent_obs_bw = recent_obs
            self.recent_dones_bw = recent_dones
        return np.array(new_obs)

    def forward(self, sess, ob, done, out_type='pv'):
        ob = self._recent_ob(np.array([ob]), np.array([done]))
        outs = self._get_forward_outs(out_type)
        out_values = sess.run(outs, {self.obs:ob})
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        obs = self._recent_ob(np.array(obs), np.array(dones),  ob_type='backward')
        summary, _ = sess.run([self.summary, self._train],
                              {self.obs:obs,
                               self.A:acts,
                               self.ADV:Advs,
                               self.R:Rs,
                               self.lr:cur_lr,
                               self.entropy_coef:cur_beta})
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step=global_step)


class FcPolicy(Policy):
    def __init__(self, n_s, n_a, n_step, n_fc0=256, n_fc=128, name=None):
        super().__init__(n_a, n_s, n_step, 'fc', name)
        self.n_fc = n_fc0
        self.obs = tf.placeholder(tf.float32, [None, n_s])
        with tf.variable_scope(self.name):
            # pi and v use separate nets
            self.pi = self._build_net(n_fc, 'pi')
            self.v = self._build_net(n_fc, 'v')

    def _build_net(self, n_fc, out_type):
        h = fc(self.obs, out_type + '_fc0', self.n_fc)
        return self._build_fc_net(h, n_fc, out_type)

    def forward(self, sess, ob, done, out_type='pv'):
        outs = self._get_forward_outs(out_type)
        out_values = sess.run(outs, {self.obs:[ob]})
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                 summary_writer=None, global_step=None):
        summary, _ = sess.run([self.summary, self._train],
                              {self.obs:obs,
                               self.A:acts,
                               self.ADV:Advs,
                               self.R:Rs,
                               self.lr:cur_lr,
                               self.entropy_coef:cur_beta})
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step=global_step)
