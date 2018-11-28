import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


class Normalizer():
    def __init__(self, input_size):
        self.n = torch.zeros(input_size)
        self.mean = torch.zeros(input_size)
        self.mean_diff = torch.zeros(input_size)
        self.var = torch.zeros(input_size)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x - last_mean)*(x - self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std


class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size, bias=False)
        self.affine2 = nn.Linear(hidden_size, output_size, bias=False)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        action_scores = F.softmax(x, dim=0)
        return action_scores


def select_action(policy, state):
    probs = policy(Variable(state))
    m = Categorical(probs)
    selected_action = m.sample()
    action = torch.Tensor([0, 0])
    action[selected_action.data] = 1
    log_prob = m.log_prob(selected_action)
    policy.saved_log_probs.append(log_prob)
    return action


def finish_episode(policy, optimizer):
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    eps = np.finfo(np.float32).eps.item()
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy_loss
