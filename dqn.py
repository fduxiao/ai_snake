"""
This defines the basic setup for a DQN problem.
"""

from contextlib import contextmanager
import math
from collections import namedtuple
import random

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


def next_epsilon(eps_start, eps_end, decay_step, decay):
    eps_delta = eps_start - eps_end
    eps_threshold = eps_end + eps_delta * decay
    # the same as the following:
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #         math.exp(-1. * steps_done / EPS_DECAY)
    decay *= math.exp(-1. / decay_step)
    return eps_threshold, decay


class DQNTrainer:

    def __init__(
            self,
            policy_net,
            action_range,
            batch_size=128,
            gamma=0.999,
            eps_start=0.9,
            eps_end=0.05,
            decay_step=200,
            target_update=20,
            capacity=10000,
            optimizer=None,
    ):
        self.policy_net = policy_net
        self.previous_dict = policy_net.state_dict()
        # hyper-arguments
        self.action_range = action_range
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_step = decay_step
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

    def save_trainer(self, dump):
        dump(self.decay)
        dump(self.memory)

    def load_trainer(self, load):
        self.decay = load()
        self.memory = load()

    @property
    def total_steps(self):
        return int(-math.log(self.decay) * self.decay_step)

    def next_epsilon(self):
        eps_threshold, self.decay = next_epsilon(self.eps_start, self.eps_end, self.decay_step, self.decay)
        return eps_threshold

    def select_action(self, state, epsilon=None):

        sample = random.random()
        if epsilon is None:
            eps_threshold = self.next_epsilon()
        else:
            eps_threshold = epsilon
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # noinspection PyCallingNonCallable,PyUnresolvedReferences,PyTypeChecker
            return torch.tensor([[random.randrange(self.action_range)]], dtype=torch.long)

    @contextmanager
    def prev_net(self):
        current_dict = self.policy_net.state_dict()
        self.policy_net.load_state_dict(self.previous_dict)
        yield
        self.policy_net.load_state_dict(current_dict)

    # noinspection PyCallingNonCallable, PyUnresolvedReferences
    def train_step(self):
        self.policy_net.train()
        self.target_counter = (self.target_counter + 1) % self.target_update
        if self.target_counter == 0:
            self.previous_dict = self.policy_net.state_dict()

        if len(self.memory) < self.batch_size:
            return "insufficient memory {}".format(len(self.memory))
        while True:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            mask_list = [
                s for s in batch.next_state if s is not None
            ]
            if len(mask_list) == 0:
                continue
            break

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(
                lambda s: s is not None, batch.next_state
            )),
            dtype=torch.uint8
        )
        non_final_next_states = torch.cat(mask_list)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # noinspection PyCallingNonCallable,PyUnresolvedReferences
        next_state_values = torch.zeros(self.batch_size)
        with self.prev_net():
            next_state_values[non_final_mask] = self.policy_net(non_final_next_states).max(1)[0].detach()
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
        return loss.item()

    def __call__(self):
        return self.train_step()
