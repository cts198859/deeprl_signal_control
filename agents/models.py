import numpy as np
import os
import tensorflow as tf
from agents.utils import *
from agents.policies import *


class A2C:
    def __init__(self, sess, n_s, n_a, total_step, i_thread=-1, model_config=None):
        self.name = 'A2C_' + str(i_thread)
        policy = model_config.get('POLICY')
        v_coef = model_config.getfloat('VALUE_COEF')
        max_grad_norm = model_config.getfloat('MAX_GRAD_NORM')
        alpha = model_config.getfloat('RMSP_ALPHA')
        epsilon = model_config.getfloat('RMSP_EPSILON')
        lr_init = model_config.getfloat('LR_INIT')
        lr_min = model_config.getfloat('LR_MIN')
        lr_decay = model_config.get('LR_DECAY')
        beta_init = model_config.getfloat('ENTROPY_COEF_INIT')
        beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
        beta_decay = model_config.get('ENTROPY_DECAY')
        beta_ratio = model_config.getfloat('ENTROPY_RATIO')
        gamma = model_config.getfloat('GAMMA')
        n_step = model_config.getint('NUM_STEP')
        n_past = model_config.getint('NUM_PAST')

        if policy == 'lstm':
            n_lstm = model_config.getint('NUM_LSTM')
            n_fc = model_config.get('NUM_FC').split(',')
            n_fc = [int(x) for x in n_fc]
            self.policy = LstmPolicy(n_s, n_a, n_step, n_past, i_thread, n_fc=n_fc, n_lstm=n_lstm)
        self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)
        sess.run(tf.global_variables_initializer())

        if i_thread == -1:
            self.lr_scheduler = Scheduler(lr_init, lr_min, total_step, decay=lr_decay)
            self.beta_scheduler = Scheduler(beta_init, beta_min, total_step * beta_ratio,
                                            decay=beta_decay)
        self.trans_buffer = OnPolicyBuffer(gamma)

        def save(saver, model_dir, global_step):
            if i_thread == -1:
                saver.save(sess, model_dir, global_step=global_step)

        def load(saver, model_dir):
            if i_thread == -1:
                save_file = None
                save_step = 0
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.startswith('step'):
                            prefix = file.split('.')[0]
                            cur_step = int(prefix.split('-')[1])
                            if cur_step > save_step:
                                save_file = prefix
                                save_step = cur_step
                if save_file is not None:
                    saver.restore(sess, model_dir + save_file)
                    print('checkpoint loaded: ', save_file)
                else:
                    print('could not find old checkpoint')

        def backward(R, cur_lr=None, cur_beta=None, train=True):
            obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
            if cur_lr is None:
                cur_lr = self.lr_scheduler.get(n_step)
            if cur_beta is None:
                cur_beta = self.beta_scheduler.get(n_step)
            return self.policy.backward(sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta, train)

        def forward(ob, done, out_type='pv'):
            return self.policy.forward(sess, ob, done, out_type)

        self.save = save
        self.load = load
        self.backward = backward
        self.forward = forward
        self.n_step = n_step
        self.add_transition = self.trans_buffer.add_transition
