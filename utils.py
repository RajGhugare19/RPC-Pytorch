import numpy as np
import torch.nn as nn
from typing import Iterable

class ReplayMemory():
    def __init__(self, buffer_limit, obs_size, action_size, obs_dtype):
        self.buffer_limit = buffer_limit
        self.observation = np.empty((buffer_limit, obs_size), dtype=obs_dtype) 
        self.next_observation = np.empty((buffer_limit, obs_size), dtype=obs_dtype) 
        self.action = np.empty((buffer_limit, action_size), dtype=np.float32)
        self.reward = np.empty((buffer_limit,), dtype=np.float32) 
        self.terminal = np.empty((buffer_limit,), dtype=bool)
        self.idx = 0
        self.full = False

    def push(self, transition):
        state, action, reward, next_state, done = transition
        self.observation[self.idx] = state
        self.next_observation[self.idx] = next_state
        self.action[self.idx] = action 
        self.reward[self.idx] = reward
        self.terminal[self.idx] = done
        self.idx = (self.idx + 1) % self.buffer_limit
        self.full = self.full or self.idx == 0
    
    def push_batch(self, transition_batch, batch_size):
        state, action, reward, next_state, done = transition_batch
        batch_idx = np.arange(batch_size)%self.buffer_limit
        self.observation[batch_idx] = state
        self.next_observation[batch_idx] = next_state
        self.action[batch_idx] = action 
        self.reward[batch_idx] = reward
        self.terminal[batch_idx] = done

        self.full = self.full or self.idx+batch_size >= self.buffer_limit
        self.idx = (self.idx + batch_size) % self.buffer_limit

    def sample(self, n):
        idxes = np.random.randint(0, self.buffer_limit if self.full else self.idx, size=n)
        return self.observation[idxes], self.action[idxes], self.reward[idxes], self.next_observation[idxes], self.terminal[idxes]

    def sample_all(self):
        idx = self.buffer_limit if self.full else self.idx+1
        return self.observation[:idx], self.action[:idx], self.reward[:idx], self.next_observation[:idx], self.terminal[:idx]

    def _sample_idx(self, L):
        valid_idx = False 
        while not valid_idx:
            idx = np.random.randint(0, self.buffer_limit if self.full else self.idx-L)
            idxs = np.arange(idx, idx+L)%self.buffer_limit
            valid_idx = not self.idx in idxs[1:]
        return idxs 

    def _retrieve_batch(self, idxs, n, l):
        vec_idxs = idxs.transpose().reshape(-1)
        return self.observation[vec_idxs].reshape(l, n, -1), self.action[vec_idxs].reshape(l, n, -1), self.reward[vec_idxs].reshape(l, n), \
            self.next_observation[vec_idxs].reshape(l, n, -1), self.terminal[vec_idxs].reshape(l, n)
    
    def sample_seq(self, seq_len, batch_size):
        n = batch_size
        l = seq_len
        obs, act, rew, next_obs, term = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        return obs, act, rew, next_obs, term

    def __len__(self):
        return self.buffer_limit if self.full else self.idx+1

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def get_parameters(modules: Iterable[nn.Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]