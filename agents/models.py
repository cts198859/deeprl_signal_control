import os
from agents.utils import *
from agents.policies import *


class A2C:
    def __init__(self, n_s, n_a, total_step, model_config, seed=0):
        # load parameters
        self.n_agent = 1
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_s = n_s
        self.n_a = n_a
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self._init_policy(model_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self._init_train(total_step, model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, model_config):
        n_step = model_config.getint('batch_size')
        n_h = model_config.getint('num_h')
        policy_name = model_config.get('policy')
        if policy_name == 'lstm':
            n_lstm = model_config.getint('num_lstm')
            policy = LstmPolicy(self.n_s, self.n_a, n_step, n_fc=n_h, n_lstm=n_lstm)
        elif policy_name == 'cnn1':
            n_filter = model_config.getint('num_filter')
            m_filter = model_config.getint('size_filter')
            n_past = model_config.getint('num_past')
            policy = Cnn1DPolicy(self.n_s, self.n_a, n_step, n_past, n_fc=n_h,
                                 n_filter=n_filter, m_filter=m_filter)
        elif policy_name == 'fc':
            n_fc = model_config.getint('num_fc')
            policy = FcPolicy(self.n_s, self.n_a, n_step, n_fc0=n_fc, n_fc=n_h)
        self.policy = policy
        self.n_step = n_step

    def _init_train(self, total_step, model_config):
        # init loss
        v_coef = model_config.getfloat('value_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.policy.prepare_loss(v_coef, max_grad_norm, alpha, epsilon)

        # init reward norm/clip
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')

        # init scheduler
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        beta_init = model_config.getfloat('entropy_coef_init')
        beta_decay = model_config.get('entropy_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, total_step, decay=lr_decay)
        if beta_decay == 'constant':
            self.beta_scheduler = Scheduler(beta_init, decay=beta_decay)
        else:
            beta_min = model_config.getfloat('ENTROPY_COEF_MIN')
            beta_ratio = model_config.getfloat('ENTROPY_RATIO')
            self.beta_scheduler = Scheduler(beta_init, beta_min, total_step * beta_ratio,
                                            decay=beta_decay)

        # init replay buffer
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = OnPolicyBuffer(gamma)

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(sess, model_dir + save_file)
            print('checkpoint loaded: ', save_file)
        else:
            print('could not find old checkpoint')

    def backward(self, R, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        cur_beta = self.beta_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta,
                             summary_writer=summary_writer, global_step=global_step)

    def forward(self, ob, done, out_type='pv'):
        return self.policy.forward(self.sess, ob, done, out_type)

    def add_transition(self, ob, action, reward, value, done):
        if self.reward_norm:
            reward /= self.reward_norm
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        self.trans_buffer.add_transition(ob, action, reward, value, done)


class MultiA2C(A2C):
    def __init__(self, sess, n_s_ls, n_a_ls, total_step, i_thread=-1, optimizer_ls=None, lr_ls=None,
                 model_config=None, discrete=True):
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.i_thread = i_thread
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
        n_fc = model_config.getint('NUM_FC')
        self.reward_norm = model_config.getfloat('REWARD_NORM')
        self.lrs = []
        self.optimizers = []
        self.names = []
        self.trans_buffers = []
        for i in range(self.n_agent):
            n_s = n_s_ls[i]
            n_a = n_a_ls[i]
            name = 'agent' + str(i)
            if policy == 'lstm':
                n_lstm = model_config.getint('NUM_LSTM')
                policy = LstmPolicy(n_s, n_a, n_step, i_thread, n_past, n_fc=n_fc,
                                    n_lstm=n_lstm, discrete=discrete, name=name)
            elif policy == 'cnn1':
                n_filter = model_config.getint('NUM_FILTER')
                m_filter = model_config.getint('SIZE_FILTER')
                policy = Cnn1DPolicy(n_s, n_a, n_step, i_thread, n_past,
                                     n_fc=n_fc, n_filter=n_filter,
                                     m_filter=m_filter, discrete=discrete, name=name)
            self.agents.append(policy)
            self.names.append(policy.name)
            if i_thread == -1:
                policy.prepare_loss(None, None, v_coef, max_grad_norm, alpha, epsilon)
                self.lrs.append(policy.lr)
                self.optimizers.append(policy.optimizer)
            self.trans_buffers.append(OnPolicyBuffer(gamma))

        if (i_thread == -1) and (total_step > 0):
            # global lr and entropy beta scheduler
            self.lr_scheduler = Scheduler(lr_init, lr_min, total_step, decay=lr_decay)
            self.beta_scheduler = Scheduler(beta_init, beta_min, total_step * beta_ratio,
                                            decay=beta_decay)
            self.saver = tf.train.Saver(max_to_keep=20)

    def backward(self, sess, R_ls, cur_lr, cur_beta):
        assert (self.i_thread >= 0), 'incorrect update on global network!'
        discrete = self.agents[0].discrete
        for i in range(self.n_agent):
            obs, acts, dones, Rs, Advs = self.trans_buffers[i].sample_transition(R_ls[i], discrete)
            self.agents[i].backward(sess, obs, acts, dones, Rs, Advs, cur_lr, cur_beta)

    def forward(self, i_agent, sess, ob, done, out_type='pv'):
        assert (self.i_thread >= 0), 'cannot explore with global network!'
        return self.agents[i_agent].forward(sess, ob, done, out_type)

    def add_transition(self, i_agent, ob, action, reward, value, done):
        if self.reward_norm:
            reward /= self.reward_norm
        self.trans_buffers[i_agent].add_transition(ob, action, reward, value, done)

