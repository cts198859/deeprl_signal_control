import gym
import numpy as np


class GymEnv:
    def __init__(self, name):
        env = gym.make(name)
        self.n_a = env.action_space.n
        self.n_a_ls = [self.n_a]
        self.n_s = env.observation_space.shape[0]
        self.n_s_ls = [self.n_s]
        s_min = np.array(env.observation_space.low)
        s_max = np.array(env.observation_space.high)
        s_mean, s_scale = .5 * (s_min + s_max), .5 * (s_max - s_min)

        def scale_ob(ob):
            return (np.array(ob) - s_mean) / s_scale

        def reset():
            return [scale_ob(env.reset())]

        def step(action):
            ob, r, done, info = env.step(action[0])
            return [scale_ob(np.ravel(ob))], [r], done, [action], r

        self.seed = env.seed
        self.step = step
        self.reset = reset