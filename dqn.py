"""
This defines the basic setup for a DQN problem.
"""

import math
import random
from collections import namedtuple

import torch
import torch.optim as optim
import torch.nn.functional as functional

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNTrainer:

    def __init__(
            self,
            policy_net,
            target_net,
            action_range,
            batch_size=128,
            gamma=0.999,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=200,
            target_update=10,
            capacity=10000,
            optimizer=None
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        # hyper-arguments
        self.action_range = action_range
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        # used to calculate decay
        self.decay = 1
        # target counter
        self.target_counter = 0
        # memory
        self.memory = ReplayMemory(capacity)
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(policy_net.parameters())

    def select_action(self, state):

        sample = random.random()
        eps_delta = self.eps_start - self.eps_end
        eps_threshold = self.eps_end + eps_delta * self.decay
        # the same as the following:
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #         math.exp(-1. * steps_done / EPS_DECAY)
        self.decay *= math.exp(-1. / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # noinspection PyCallingNonCallable,PyUnresolvedReferences,PyTypeChecker
            return torch.tensor([[random.randrange(self.action_range)]], dtype=torch.long)

    # noinspection PyCallingNonCallable, PyUnresolvedReferences
    def train_step(self):
        # self.target_net.train()
        # self.policy_net.train()
        self.target_counter = (self.target_counter + 1) % self.target_update
        if self.target_counter == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(
                lambda s: s is not None, batch.next_state
            )),
            dtype=torch.uint8
        )
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None
        ])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # noinspection PyCallingNonCallable,PyUnresolvedReferences
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def __call__(self):
        self.train_step()
