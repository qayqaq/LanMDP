import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import gym
# import d4rl

def get_exp_dataset(env_name='hopper', num=5):
    env = gym.make(env_name + '-expert-v2')
    dataset = env.get_dataset('exp2/hdf5/' + env_name + '_expert-v2.hdf5')

    obs = dataset["observations"]
    act = dataset["actions"]
    rwd = dataset["rewards"].reshape(-1, 1)
    next_obs = dataset["next_observations"]
    terminated = dataset["terminals"]
    timeout = dataset["timeouts"]
    done = terminated | timeout

    end = done.nonzero()[0][num] + 1


    dataset = {'state': torch.from_numpy(obs[:end]),
            'action': torch.from_numpy(act[:end]),
            'reward': torch.from_numpy(rwd[:end]),
            'next_state': torch.from_numpy(next_obs[:end]),
            'done': torch.from_numpy(done[:end])}
    
    print('expert average return: ', dataset['reward'].sum() / num)
    torch.save(dataset, 'new-hopper-demo.pth')
    
    return dataset


def get_imp_dataset(env_name='hopper', num=100):
    traj_obs = []
    traj_act = []
    traj_rwd = []
    traj_next_obs = []
    traj_done = []
    kinds = ['expert', 'medium', 'random']
    
    for kind in kinds:

        env = gym.make(env_name + '-' + kind + '-v2')
        dataset = env.get_dataset('exp2/hdf5/' + env_name + '_' + kind + '-v2.hdf5')

        obs = dataset["observations"]
        act = dataset["actions"]
        rwd = dataset["rewards"].reshape(-1, 1)
        next_obs = dataset["next_observations"]
        terminated = dataset["terminals"].reshape(-1, 1)
        timeout = dataset["timeouts"].reshape(-1, 1)
        done = terminated | timeout

        print("data size:", obs.shape, act.shape, rwd.shape, next_obs.shape, terminated.shape, timeout.shape)
        print("num terminated:", sum(terminated))
        print("num timeout:", sum(timeout))
        print("num terminated and timeout:", sum(done))

        print(done.nonzero())
        tttt = np.concatenate([[-1], done.nonzero()[0]])
        idx = np.sort(np.random.choice(np.arange(tttt.size), num))
        idx = np.arange(num)
        for i in range(idx.size):
            start = tttt[idx[i]] + 1
            end = tttt[idx[i]+1]
            traj_obs.append(obs[start:end])
            traj_act.append(act[start:end])
            traj_rwd.append(rwd[start:end])
            traj_next_obs.append(next_obs[start:end])
            traj_done.append(done[start:end])

    traj_obs = np.vstack(traj_obs)
    traj_act = np.vstack(traj_act)
    traj_rwd = np.vstack(traj_rwd)
    traj_next_obs = np.vstack(traj_next_obs)
    traj_done = np.vstack(traj_done)
    traj_delta = traj_next_obs - traj_obs

    traj_obs_mean = traj_obs.mean(axis=0)
    traj_act_mean = traj_act.mean(axis=0)
    traj_next_obs_mean = traj_next_obs.mean(axis=0)
    traj_delta_mean = traj_delta.mean(axis=0)
    traj_obs_std = traj_obs.std(axis=0)
    traj_act_std = traj_act.std(axis=0)
    traj_next_obs_std = traj_next_obs.std(axis=0)
    traj_delta_std = traj_delta.std(axis=0)
    traj_obs_norm = (traj_obs - traj_obs_mean) / traj_obs_std
    traj_act_norm = (traj_act - traj_act_mean) / traj_act_std
    traj_next_obs_norm = (traj_next_obs - traj_next_obs_mean) / traj_next_obs_std
    traj_delta_norm = (traj_delta - traj_delta_mean) / traj_delta_std


    # traj_obs = torch.from_numpy(traj_obs)
    # traj_act = torch.from_numpy(traj_act)
    # traj_rwd = torch.from_numpy(traj_rwd)
    # traj_next_obs = torch.from_numpy(traj_next_obs)
    # traj_done = torch.from_numpy(traj_done)

    dataset = {'state': torch.from_numpy(traj_obs),
            'action': torch.from_numpy(traj_act),
            'next_state': torch.from_numpy(traj_next_obs),
            'delta': torch.from_numpy(traj_delta),
            'state_norm': torch.from_numpy(traj_obs_norm),
            'action_norm': torch.from_numpy(traj_act_norm),
            'next_state_norm': torch.from_numpy(traj_next_obs_norm),
            'delta_norm': torch.from_numpy(traj_delta_norm),
            'reward': torch.from_numpy(traj_rwd),
            'done': torch.from_numpy(traj_done),
            'state_mean': torch.from_numpy(traj_obs_mean),
            'action_mean': torch.from_numpy(traj_act_mean),
            'next_state_mean': torch.from_numpy(traj_next_obs_mean),
            'delta_mean': torch.from_numpy(traj_delta_mean),
            'state_std': torch.from_numpy(traj_obs_std),
            'action_std': torch.from_numpy(traj_act_std),
            'next_state_std': torch.from_numpy(traj_next_obs_std),
            'delta_std': torch.from_numpy(traj_delta_std)}
    
    return dataset

# get_imp_dataset()


class transition_dataset(Dataset):
    def __init__(self, transitions):
        self.data = []        
        for transition in transitions:
            s, a, s_prime = transition
            self.data.append([s, a, s_prime]) 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s, a, s_prime = self.data[idx]
        return s, a, s_prime




class TrainingSet(Dataset):
    def __init__(self, rawdata, env, context_len):
        super().__init__()
        self.context_len = context_len
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        data_states = rawdata['state']
        self.len = data_states.shape[0]
        data_actions = rawdata['action']
        data_rewards = rawdata['reward']
        data_dones = rawdata['done']
        data_next_states = rawdata['next_state']
        data_done_idx = data_dones.squeeze(-1).nonzero().squeeze(-1)
        self.data_done_idx = data_done_idx
        self.states = torch.cat((torch.zeros(context_len - 1, data_states.shape[1]), data_states, data_next_states[-1:,:]), dim=0)
        self.actions = torch.cat((torch.zeros(context_len - 1, data_actions.shape[1]), data_actions, torch.zeros(1, data_actions.shape[1])), dim=0)
        self.next_states = data_next_states

        sns = torch.cat((data_states, data_next_states), dim=1)
        self.normalized_sns = torch.sub(torch.mul(torch.div(torch.sub(sns, sns.min()), torch.sub(sns.max(), sns.min())), 2), 1)  # [-1, 1]

        # max_timestep = max(data_done_idx[0]+1, max(data_done_idx[1:] - data_done_idx[:-1]))
        # data_states = torch.cat(self.pad(list(torch.tensor_split(data_states, data_done_idx[:-1]+1)), max_timestep), dim=0)
        # data_actions = torch.cat(self.pad(list(torch.tensor_split(data_actions, data_done_idx[:-1]+1)), max_timestep), dim=0)
        # data_dones = torch.cat(self.pad(list(torch.tensor_split(data_dones, data_done_idx[:-1]+1)), max_timestep), dim=0)

        # num, max_timestep, _ = data_states.shape
        # timesteps = torch.arange(start=1, end=max_timestep+1, step=1)
        # timesteps = timesteps.repeat(num, 1).unsqueeze(-1)
        # data_with_time = torch.cat((timesteps, data_states), dim=-1)
        # data_time = []
        # data_state = []
        # data_action = []
        # for i in range(num):
        #     data_without_pad = data_with_time[i, :, :]
        #     timesteps = data_without_pad[:, 0:1]
        #     states = data_without_pad[:, 1:]
        #     actions = data_actions[i]
        #     length = actions.shape[0]
        #     for t in range(context_len - 1):
        #         time = torch.zeros((1, context_len+1, 1), dtype=torch.int32)
        #         s = torch.zeros((1, context_len+1, states.shape[1]))
        #         a = torch.zeros((1, context_len+1, actions.shape[1]))
        #         time[0, -t-2:, :] = timesteps[:t+2, :].unsqueeze(0)
        #         s[0, -t-2:, :] = states[:t+2, :].unsqueeze(0)
        #         a[0, -t-2:, :] = actions[:t+2, :].unsqueeze(0)
        #         data_time.append(time)
        #         data_state.append(s)
        #         data_action.append(a)
        #     for t in range(length - context_len):
        #         data_time.append(timesteps[t:t+context_len+1, :].unsqueeze(0))
        #         data_state.append(states[t:t+context_len+1, :].unsqueeze(0))
        #         data_action.append(actions[t:t+context_len+1, :].unsqueeze(0))
        # data_time = torch.vstack(data_time)
        # data_state = torch.vstack(data_state)
        # data_action = torch.vstack(data_action)
        # data = torch.cat((data_time, data_state, data_action), dim=-1)
        # idx = torch.randperm(data.shape[0])
        # self.data = data[idx].view(data.size())
        print('training data loaded')
    
    def __getitem__(self, index):
        index += self.context_len - 1
        ll = index - self.data_done_idx
        pad = ((ll >= 1) & (ll < self.context_len)).nonzero().size()[0]
        s = torch.cat((self.states[index - self.context_len + 1 : index + 1].float(), self.next_states[index].unsqueeze(0)), dim=0)
        a = self.actions[index - self.context_len + 1 : index + 2].float()
        if pad > 0:
            s[:pad] = 0.
            a[:pad] = 0.

        return 0, s, a

        return 0, self.states[index - self.context_len + 1 : index + 2].float(), self.actions[index - self.context_len + 1 : index + 2].float()
        return self.data[index,:,0].to(torch.int32), self.data[index,:,1:1+self.state_dim].to(torch.float32), self.data[index,:,1+self.state_dim:1+self.state_dim+self.action_dim].to(torch.float32)

    def __len__(self):
        return self.len
        return self.data.shape[0]
    
    def pad(self, x, max_timestep, pad=0):
        for i in range(len(x)):
            d = x[i]
            zeros = torch.zeros((max_timestep - len(d), d.shape[1]))
            x[i] = torch.cat((d, zeros), dim=0).unsqueeze(0)
        return x