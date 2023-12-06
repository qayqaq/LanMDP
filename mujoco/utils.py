import torch
from torch.distributions import Categorical, Normal
import numpy as np

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def select_action_continuous(state, policy):
	state = torch.from_numpy(state).float()
	mean, sigma = policy(state)
	sigma = 0.1
	pi = Normal(loc=mean, scale=sigma)
	action = pi.sample()
	return action.numpy()

def select_action_discrete(state, policy):
	state = torch.from_numpy(state).float()
	probs = policy(state)
	pi = Categorical(probs)
	action = pi.sample()
	return action.numpy()
