import numpy as np
from gym import logger

from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from collections import namedtuple

class Normalizer():
    def __init__(self, input_size):
        self.n = torch.zeros(input_size)
        self.mean = torch.zeros(input_size)
        self.mean_diff = torch.zeros(input_size)
        self.var = torch.zeros(input_size)

    def _observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x - last_mean)*(x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def _normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean) / obs_std

class RandomAgent(object):
    """Random agent."""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state, reward, done):
        return self.action_space.sample()

class ReinforceAgent(nn.Module):
    """Reinforce agent."""
    def __init__(self, input_size, hidden_size, output_size,
        learning_rate, gamma):
        super(ReinforceAgent, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size, bias=False)
        self.affine2 = nn.Linear(hidden_size, output_size, bias=False)
        self.saved_log_probs = []
        self.rewards = []

        # build in an optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.normalizer = Normalizer(input_size=input_size)
        self.gamma = gamma


    def _forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        action_scores = F.softmax(x, dim=0)
        return action_scores

    def act(self, state, reward, done):
        """Return the action, finish episode if done."""
        # add the reward to the accumulator
        self.rewards.append(reward)
        if done:
            self._update_agent()

        # Normalize the state
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float()
        self.normalizer._observe(state)
        state = self.normalizer._normalize(state)
        action_temp = self._select_action(state)
        return action_temp.data[0]

    def _select_action(self, state):
        """Select the action based on the current policy."""
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float()
        probs = self._forward(Variable(state))
        m = Categorical(probs)
        selected_action = m.sample()
        log_prob = m.log_prob(selected_action)
        self.saved_log_probs.append(log_prob)
        action = selected_action
        return action

    def _update_agent(self):
        """When episode finishes, calculate policy loss and update model."""
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        logger.debug('Policy loss: {}'.format(policy_loss.data[0]))

class ActorCriticAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
        learning_rate, gamma):
        super(ActorCriticAgent, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size, bias=False)
        self.action_head = nn.Linear(hidden_size, output_size, bias=False)
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        self.saved_actions = []
        self.rewards = []
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

        # build in an optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.normalizer = Normalizer(input_size=input_size)
        self.gamma = gamma

    def _forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def act(self, state, reward, done):
        """Return the action, finish episode if done."""
        # add the reward to the accumulator
        self.rewards.append(reward)
        if done:
            self._update_agent()

        # Normalize the state
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float()
        self.normalizer._observe(state)
        state = self.normalizer._normalize(state)
        action_temp = self._select_action(state)
        return action_temp.data[0]

    def _select_action(self, state):
        """Select the action based on the current policy."""
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float()
        probs, state_value = self._forward(Variable(state))
        m = Categorical(probs)
        selected_action = m.sample()
        log_prob = m.log_prob(selected_action)
        self.saved_actions.append(self.SavedAction(log_prob, state_value))
        action = selected_action
        return action

    def _update_agent(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.data[0]
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]