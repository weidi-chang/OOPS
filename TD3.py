import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
import copy
from torch.nn.utils import clip_grad_norm_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action


    def forward(self, x, t):
        xt = torch.cat([x, t], 1)
        xt = torch.relu(self.l1(xt))
        xt = torch.relu(self.l2(xt))
        xt = self.max_action * torch.tanh(self.l3(xt))
        return xt


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 1024)
        self.l5 = nn.Linear(1024, 1024)
        self.l6 = nn.Linear(1024, 1)


    def forward(self, x, u,t):
        xu = torch.cat([x, u, t], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = F.relu(self.l3(x1))

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = F.relu(self.l6(x2))
        return x1, x2


    def Q1(self, x, u, t):
        xu = torch.cat([x, u, t], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = F.relu(self.l3(x1))
        return x1


class Critic2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_hiddens=1, use_match=False):
        super(Critic2, self).__init__()

        self.use_match = use_match
        self.Q_1 = nn.ModuleList()
        self.Q_2 = nn.ModuleList()
        if not use_match:
            self.Q_1.append(nn.Linear(state_dim + action_dim, hidden_dim))
            self.Q_2.append(nn.Linear(state_dim + action_dim, hidden_dim))
        else:
            self.Q_1.append(nn.Linear(state_dim + action_dim+1, hidden_dim))
            self.Q_2.append(nn.Linear(state_dim + action_dim+1, hidden_dim))

        for i in range(num_hiddens):
            self.Q_1.append(nn.Linear(hidden_dim, hidden_dim))
            self.Q_2.append(nn.Linear(hidden_dim, hidden_dim))

        self.Q_1.append(nn.Linear(hidden_dim, 1))
        self.Q_2.append(nn.Linear(hidden_dim, 1))

    def forward(self, x, u, t, match):
        if self.use_match:
            xu = torch.cat([x, u, t, match], 1)
        else:
            xu = torch.cat([x, u, t], 1)

        x1 = xu
        x2 = xu

        for layer in self.Q_1[:-1]:
            x1 = F.relu(layer(x1))
        for layer in self.Q_2[:-1]:
            x2 = F.relu(layer(x2))

        x1 = self.Q_1[-1](x1)
        x2 = self.Q_2[-1](x2)
        return x1, x2

    def Q1(self, x, u, t, match):
        if self.use_match:
            xu = torch.cat([x, u, t, match], 1)
        else:
            xu = torch.cat([x, u, t], 1)
        x1 = xu
        for layer in self.Q_1[:-1]:
            x1 = F.relu(layer(x1))
        x1 = self.Q_1[-1](x1)
        return x1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, args, discount=0.99,
                tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                reward_scale=5., actor_lr=3e-4, critic_lr=3e-4, use_match=False,
                alpha=.4, min_priority=1):

        self.actor = Actor(state_dim+1, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic2(state_dim+1, action_dim, hidden_dim=1024, num_hiddens=1, use_match=use_match).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        self.reward_sigma = reward_scale * 1000. / np.sqrt(state_dim)
        self.reward_scale = reward_scale
        self.first = True
        self.use_match = use_match

        self.state_mean = torch.zeros((1, state_dim)).to(device)
        self.state_std = torch.ones((1, state_dim)).to(device)

        self.t_std = 1.
        self.aug_time = args.aug_time

        self.critic_clip = args.critic_clip
        self.actor_clip = args.actor_clip

        self.alpha = alpha
        self.min_priority = min_priority

    def select_action(self, state, t):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        s_normed = self.normz(state)
        t = torch.FloatTensor([t/1000.]).reshape(1, -1).to(device)

        return self.actor(s_normed, t).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=100, iterations=1):
        if self.first:
            buffer_rewards = replay_buffer.reward[:replay_buffer.size]
            buffer_match = replay_buffer.match[:replay_buffer.size]
            buffer_t = replay_buffer.t_to_horizon[:replay_buffer.size]

            self.rwd_std = buffer_rewards.std()
            self.match_std = buffer_match.std()
            self.t_std = buffer_t.std()

            self.rwd_mean = buffer_rewards.mean()
            print(self.rwd_mean)
            print(self.rwd_std)
            self.first = False

        for it in range(iterations):
            self.total_it += 1
            # Sample replay buffer
            state, action, next_state, reward, not_done, t, match = replay_buffer.sample(batch_size)
            match_normed = match/1000.
            next_matched_normed = (match+1.)/1000.
            s_normed = self.normz(state)
            ns_normed = self.normz(next_state)

            not_done = ~(t==1)
            if self.aug_time:
                t_H_plus = torch.LongTensor(np.random.randint(1, 1001, t.shape)).to(device)
            else:
                t_H_plus = t

            t_to_horizon = (t_H_plus)/1000.
            next_t_to_horizon = (t_H_plus-1.)/1000.
            t_H_normed = (t/1000.)
            not_done = t_H_plus > 1.

            reward = self.reward_sigma*reward

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action = (
                    self.actor_target(ns_normed, next_t_to_horizon) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(ns_normed, next_action, next_t_to_horizon, next_matched_normed)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(s_normed, action, t_to_horizon, match_normed)

            # Compute critic loss
            td_loss1 = (current_Q1 - target_Q)
            td_loss2 = (current_Q2 - target_Q)

            critic_loss = self.PAL(td_loss1) + self.PAL(td_loss2)
            critic_loss /= torch.max(td_loss1.abs(), td_loss2.abs()).clamp(min=self.min_priority).pow(self.alpha).mean().detach()
            # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.critic_clip > 0.:
                clip_grad_norm_(self.critic.parameters(), self.critic_clip)

            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor losse
                actor_loss = -self.critic.Q1(s_normed, self.actor(s_normed, t_H_normed), t_H_normed, match_normed).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.actor_clip > 0.:
                    clip_grad_norm_(self.actor.parameters(), self.actor_clip)

                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # If min_priority=1, this can be simplified.
    def PAL(self, x):
        return torch.where(
            x.abs() < self.min_priority,
            (self.min_priority ** self.alpha) * 0.5 * x.pow(2),
            self.min_priority * x.abs().pow(1. + self.alpha)/(1. + self.alpha)
        ).mean()


    def set_mean_std(self, mean, std):
        self.state_mean = torch.from_numpy(mean).float().to(device)
        self.state_std = torch.from_numpy(std).float().to(device)

    def normz(self, x, diff=False):
        if not diff:
            normed = (x-self.state_mean)/(self.state_std+1e-8)
        else:
            normed = (x-self.diff_mean)/(self.diff_std+1e-8)
        return normed

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
