import numpy as np
import tensorflow as tf
from agents.utils import *
import bisect


class Policy:
    def __init__(self, n_a, n_s, n_step, n_past):
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.n_past = n_past

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _build_fc_nn(self, x, n_fc, out_type):
        
        if out_type.startswith('pi'):
            pi = fc(x, out_type, self.n_a, act=lambda x: x)
            return tf.squeeze(tf.nn.softmax(pi))
        else:
            v = fc(x, out_type, 1, act=lambda x: x)
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

    def _get_backward_outs(self, train):
        outs = [self._summary]
        if train:
            outs.append(self._train)
        else:
            # return the local grads instead
            outs.append(self.grads)
        return outs

    def _return_backward_outs(self, out_values, train):
        if train:
            return out_values[0]
        else:
            return out_values

    def prepare_loss(self, v_coef, max_grad_norm, alpha, epsilon):
        self.A = tf.placeholder(tf.int32, [self.n_step])
        self.ADV = tf.placeholder(tf.float32, [self.n_step])
        self.R = tf.placeholder(tf.float32, [self.n_step])
        self.lr = tf.placeholder(tf.float32, [])
        self.entropy_coef = tf.placeholder(tf.float32, [])
        A_sparse = tf.one_hot(self.A, self.n_a)

        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
        entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_pi * A_sparse, axis=1) * self.ADV)
        value_loss = tf.reduce_mean(tf.square(self.R - self.v)) * 0.5 * v_coef
        self.loss = policy_loss + value_loss + entropy_loss

        wts = tf.trainable_variables()
        grads = tf.gradients(self.loss, wts)
        if max_grad_norm > 0:
            grads, self.grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.grads = list(zip(grads, wts))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=alpha, epsilon=epsilon)
        self._train = self.optimizer.apply_gradients(self.grads)

        self.summaries.append(tf.summary.scalar('loss/entropy', entropy_loss))
        self.summaries.append(tf.summary.scalar('loss/policy', policy_loss))
        self.summaries.append(tf.summary.scalar('loss/value', value_loss))
        self.summaries.append(tf.summary.scalar('loss/total', self.loss))
        self.summaries.append(tf.summary.scalar('train/lr', self.lr))
        self.summaries.append(tf.summary.scalar('train/beta', self.entropy_coef))
        self.summaries.append(tf.summary.scalar('train/gradnorm', self.grad_norm))
        self._summary = tf.summary.merge(self.summaries)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_step, i_thread, n_past=-1, n_fc=[128], n_lstm=64):
        Policy.__init__(self, n_a, n_s, n_step, n_past)
        self.name = 'lstm_' + str(i_thread)
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.ob_fw = tf.placeholder(tf.float32, [1, n_s]) # forward 1-step
        self.done_fw = tf.placeholder(tf.float32, [1])
        self.ob_bw = tf.placeholder(tf.float32, [n_step, n_s]) # backward n-step
        self.done_bw = tf.placeholder(tf.float32, [n_step])
        self.states = tf.placeholder(tf.float32, [n_lstm * 2])
        self.summaries = []
        with tf.variable_scope(self.name):
            h_fw, self.new_states = lstm(self.ob_fw, self.done_fw, self.states, 'lstm')
            # only lstm layer is shared 
            h_fw = fc(h_fw, 'fc0', n_fc[0])
            for i, n_fc_cur in enumerate(n_fc[1:]):
                fc_cur = 'fc%d' % (i+1)
                h_fw = fc(h_fw, fc_cur, n_fc_cur)
            self.pi_fw = self._build_fc_nn(h_fw, n_fc, 'pi')
            self.v_fw = self._build_fc_nn(h_fw, n_fc, 'v')
        with tf.variable_scope(self.name, reuse=True):
            h, _ = lstm(self.ob_bw, self.done_bw, self.states, 'lstm')
            h = fc(h, 'fc0', n_fc[0])
            for i, n_fc_cur in enumerate(n_fc[1:]):
                fc_cur = 'fc%d' % (i+1)
                h = fc(h, fc_cur, n_fc_cur)
            self.pi = self._build_fc_nn(h, n_fc, 'pi')
            self.v = self._build_fc_nn(h, n_fc, 'v')
            lstm_wx = tf.get_variable('lstm/wx')
            lstm_wh = tf.get_variable('lstm/wh')
            fc_w = tf.get_variable('fc0/w')
        self.summaries.append(tf.summary.histogram('train/lstm_wx', tf.reshape(lstm_wx, [-1])))
        self.summaries.append(tf.summary.histogram('train/lstm_wh', tf.reshape(lstm_wh, [-1])))
        self.summaries.append(tf.summary.histogram('train/fc_w', tf.reshape(fc_w, [-1])))
        # self.summaries.append(tf.summary.histogram('train/v_w', tf.reshape(v_w, [-1])))
        self._reset()

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = np.zeros(self.n_lstm * 2, dtype=np.float32)
        self.states_bw = np.zeros(self.n_lstm * 2, dtype=np.float32)
        self.cur_step = 0

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
        if done:
            self.cur_step = 0
        self.cur_step += 1
        if (self.n_past > 0) and (self.cur_step >= self.n_past):
            self.states_fw = np.zeros(self.n_lstm * 2, dtype=np.float32)
            self.cur_step = 0
        return self._return_forward_outs(out_values)

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta, train=True):
        outs = self._get_backward_outs(train)
        out_values = sess.run(outs, {self.ob_bw:obs,
                                     self.done_bw:dones,
                                     self.states:self.states_bw,
                                     self.A:acts,
                                     self.ADV:Advs,
                                     self.R:Rs,
                                     self.lr:cur_lr,
                                     self.entropy_coef:cur_beta})
        self.states_bw = np.copy(self.states_fw)
        return self._return_backward_outs(out_values, train)

    def _backward_policy(self, sess, obs, dones):
        pi = sess.run(self.pi, {self.ob_bw:obs, self.done_bw:dones,
                                self.states:self.states_bw})
        self.states_bw = np.copy(self.states_fw)
        return pi

    def _get_forward_outs(self, out_type):
        outs = []
        if 'p' in out_type:
            outs.append(self.pi_fw)
        if 'v' in out_type:
            outs.append(self.v_fw)
        return outs


class Cnn1DPolicy(Policy):
    def __init__(self, n_s, n_a, n_step, i_thread, n_past=10,
                 n_fc=[128], n_filter=64, m_filter=3):
        Policy.__init__(self, n_a, n_s, n_step, n_past)
        self.name = 'cnn1d_' + str(i_thread)
        self.n_filter = n_filter
        self.m_filter = m_filter
        self.obs = tf.placeholder(tf.float32, [None, n_past, n_s])
        with tf.variable_scope(self.name):
            h = conv(self.obs, 'conv', n_filter, m_filter, conv_dim=1)
            n_conv_fc = np.prod([v.value for v in h.shape[1:]])
            h = tf.reshape(h, [-1, n_conv_fc])
            # only conv layer is shared
            self.pi = self._build_fc_nn(h, n_fc, 'pi')
            self.v = self._build_fc_nn(h, n_fc, 'v')
        self._reset()

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
        if ob_type == 'backward':
            print(comb_obs)
            print(comb_dones)
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

    def backward(self, sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta, train=True):
        outs = self._get_backward_outs(train)
        obs = self._recent_ob(np.array(obs), np.array(dones),  ob_type='backward')
        out_values = sess.run(outs, {self.obs:obs,
                                     self.A:acts,
                                     self.ADV:Advs,
                                     self.R:Rs,
                                     self.lr:cur_lr,
                                     self.entropy_coef:cur_beta})
        return self._return_backward_outs(out_values, train)

    def _backward_policy(self, sess, obs, dones):
        obs = self._recent_ob(np.array(obs), np.array(dones),  ob_type='backward')
        print(obs)
        return sess.run(self.pi, {self.obs:obs})


def test_forward_backward_policies(sess, policy, x, done):
    n_step = len(done)
    for i in range(n_step):
        print('forward')
        pi = policy.forward(sess, x[i], done[i], out_type='p')
        print(pi)
        print('-' * 40)
    print('backward')
    pi = policy._backward_policy(sess, x, done)
    print(pi)
    print('=' * 40)


def test_policies():
    sess = tf.Session()
    n_s, n_a, n_step, n_past = 3, 2, 5, 8
    p_lstm = LstmPolicy(n_s, n_a, n_step, n_lstm=5)
    p_cnn1 = Cnn1DPolicy(n_s, n_a, n_step, n_past)
    sess.run(tf.global_variables_initializer())
    print('=' * 16 + 'first batch' + '=' * 16)
    x = np.random.randn(n_step, n_s)
    done = np.array([0,0,0,1,0])
    x[3,:] = 0
    print('LSTM:')
    test_forward_backward_policies(sess, p_lstm, x, done)
    print('CNN1D:')
    test_forward_backward_policies(sess, p_cnn1, x, done)
    print('=' * 16 + 'second batch' + '=' * 16)
    x = np.random.randn(n_step, n_s)
    done = np.array([0,1,1,0,0])
    x[1,:] = 0
    print('LSTM:')
    test_forward_backward_policies(sess, p_lstm, x, done)
    print('CNN1D:')
    test_forward_backward_policies(sess, p_cnn1, x, done)


if __name__ == '__main__':
    test_policies()
