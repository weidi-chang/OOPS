import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, reward_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, reward_dim))
        self.not_done = np.zeros((max_size, 1))
        self.t_to_horizon = np.zeros((max_size, 1))
        # self.traj_reward = np.zeros((max_size, traj_dim))
        self.match = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done, t_H, match):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.t_to_horizon[self.ptr] = t_H
        self.match[self.ptr] = match
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)



    def sample(self, batch_size, ind=None):
        if ind is None:
            ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.t_to_horizon[ind]).to(self.device),
            torch.FloatTensor(self.match[ind]).to(self.device)
        )

class ExpertDataset(Dataset):
    def __init__(self, demos, n_experts=0, augment=False, normalize=True, clip=False, noise_levels=None):
        """
        Args:
            expert_paths (string): Path to expert npy file
        """
        self.demos = demos
        self.expert_data = []
        self.actions = []
        self.rewards = []
        self.traj_rewards = []
        self.n_timesteps = 0
        self.n_paths = 0
        self.n_experts = n_experts
        self.augment = False
        self.normalize = normalize
        self.clip = clip

        self.noise_levels = noise_levels

        if n_experts > len(self.demos) or n_experts == 0:
            self.n_experts = len(self.demos)
        # Iterate over paths and timesteps
        for path in self.demos[:self.n_experts]:
            rewards = []
            for j in range(len(path)-1):
                self.expert_data.append([path[j][0], path[j+1][0]])
                self.actions.append(path[j][1])
                self.rewards.append(path[j][2])
                rewards.append(path[j][2])
                self.n_timesteps += 1
            self.n_paths += 1
        self.expert_data = np.asarray(self.expert_data)
        self.mean = np.mean(self.expert_data, axis=0)[0]
        self.std = np.std(self.expert_data, axis=0, ddof=1)[0]
        self.max = np.max(self.expert_data, axis=0)[0]
        self.max = np.where(np.isclose(self.max, 0.), 1., self.max) # Replace 0s by 1
        self.min = np.min(self.expert_data, axis=0)[0]

        self.diff_mean = np.mean(self.actions, axis=0)
        self.diff_std = np.std(self.actions, axis=0) # Used for additive noise
        self.diff_max = np.max(self.actions, axis=0)
        self.diff_max = np.where(np.isclose(self.diff_max, 0.), 1., self.diff_max) # Replace 0s by 1
        self.diff_min = np.min(self.actions, axis=0)
        self.diff_std = np.where(np.isclose(self.diff_std, 0.), 1., self.diff_std)

        if normalize:
            self.expert_data[:, 0] = self.normz(self.expert_data[:, 0], False)
            self.expert_data[:, 1] = self.normz(self.expert_data[:, 1], False)
            # self.actions = self.normz(self.actions, True)
        self.real_std = np.std(self.expert_data, axis=0)[0]
        self.diff_real_std = np.ones(len(self.actions[0]))

        print(("Loaded %d paths for a total of %d timesteps") %(self.n_paths, self.n_timesteps))


    def __len__(self):
        return len(self.expert_data)

    def __getitem__(self, idx):
        if self.noise_levels is not None:
            return self.expert_data[idx], self.actions[idx], self.noise_levels[idx]
        else:
            return self.expert_data[idx], self.actions[idx]

    def normz(self, x, diff=False):
        if not diff:
            normed = (x-self.mean)/(3.*self.std+1e-8)
        else:
            normed = (x-self.diff_mean)/(3.*self.diff_std+1e-8)
        if self.clip:
            normed = np.clip(normed, -10., 10.)
        return normed

    def denormz(self, x, diff=False):
        if not diff:
            normed = x*(3*self.std+1e-8) + self.mean
        else:
            normed = x*(3*self.diff_std+1e-8) + self.diff_mean
        return normed
