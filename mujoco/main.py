import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

seed = 3042
pl.seed_everything(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

from model import *
from sampler import *
from data import *
from utils import *

from mjrl.utils.gym_env import GymEnv
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths, sample_data_batch

import mbil.gym_env
from mbil.utils import *
from mbil.gym_env import model_based_env
from mbil.dataset import OnlineDataset
from mbil.cost import RBFLinearCost
from mbil.dynamics_model import DynamicsEnsemble

from math import sqrt

action_delta = torch.ones(1, device='cuda') * 1.

class DEBM(pl.LightningModule):
    def __init__(self, 
        env,
        version_id=None,
        env_name=None,
        data_file=None,
        lr=3e-4, 
        batch_size=256, 
        h_dim=8, 
        depth=8, 
        context_len=10,
        n_prior_sample=1,
        prior_sampler_steps=5,
        n_prior_sample_val=32,
        prior_sampler_stepsize_init=0.5,
        prior_sampler_stepsize_init_again=0.1,
        prior_sampler_stepsize_end=1e-5,
        noise_scale=0.5,
        regularization_alpha = 1.,
        max_test_ep_len=100,
        sigma=0.1,
        num_eval_episodes=100,
        num_epoch_eval=50,
        num_epoch_collect=100,
        num_epoch_train_env=1000,
        error_bound=0.002,
        buffer_size=10000,
        select=2):
        super(DEBM, self).__init__()

        self.env = env
        self.version_id = version_id
        self.env_name = env_name
        self.data_file = data_file
        self.lr = lr
        self.batch_size = batch_size
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.context_len = context_len
        self.n_prior_sample = n_prior_sample
        self.n_prior_sample_val = n_prior_sample_val
        self.prior_sampler_steps=prior_sampler_steps
        self.prior_sampler_stepsize_init=prior_sampler_stepsize_init
        self.prior_sampler_stepsize_init_again=prior_sampler_stepsize_init_again
        self.prior_sampler_stepsize_end=prior_sampler_stepsize_end
        self.noise_scale = noise_scale
        self.regularization_alpha = regularization_alpha
        self.max_test_ep_len = max_test_ep_len
        self.sigma = torch.tensor(sigma, device='cuda')
        self.num_eval_episodes = num_eval_episodes
        self.num_epoch_eval = num_epoch_eval
        self.num_epoch_collect = num_epoch_collect
        self.num_epoch_train_env = num_epoch_train_env
        self.error_bound = error_bound
        self.buffer_size = buffer_size
        self.select = select

        self.save_hyperparameters()
        
        self.Z = (2*torch.pi) ** (self.state_dim/2) * self.sigma ** (self.state_dim)
        self.transitions = []
        self.energies = []

        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        # self.env_pdf = MultivariateNormal(loc=torch.zeros(self.state_dim).to('cuda'), covariance_matrix=torch.eye(self.state_dim).to('cuda')*self.sigma)
        
        self.energy_model = EnergyFunc(state_dim=self.state_dim, act_dim=self.act_dim, h_dim=h_dim, context_len=context_len).cuda()
        
        self.env_model_1 = dynamics_continuous(self.state_dim, 512, self.act_dim).cuda()
        # self.env_model_2 = dynamics_continuous(self.state_dim, 512, self.act_dim).cuda()
        # self.inv_model = inv_dynamics_continuous(self.state_dim, 512, self.act_dim).cuda()

        # self.env_model_1.load_state_dict(torch.load('exp2/weight/Hopper-v6-env_model_1.pth')['net'])
        # self.env_model_2.load_state_dict(torch.load('exp2/weight/Hopper-v6-env_model_2.pth')['net'])
        # self.inv_model.load_state_dict(torch.load("exp2/weight/Hopper-v6-inv_model.pth")['net'])

        self.env_opt_1 = optim.AdamW(self.env_model_1.parameters(), lr=3e-3, weight_decay=0.0001)
        # self.env_opt_2 = optim.AdamW(self.env_model_2.parameters(), lr=3e-3, weight_decay=0.0001)
        # self.inv_opt = optim.AdamW(self.inv_model.parameters(), lr=5e-4, weight_decay=0.0001)
        # self.env_opt_1.load_state_dict(torch.load("exp2/weight/Hopper-v6-env_model_1.pth")['optimizer'])
        # self.env_opt_2.load_state_dict(torch.load("exp2/weight/Hopper-v6-env_model_2.pth")['optimizer'])
        # self.inv_opt.load_state_dict(torch.load("exp2/weight/Hopper-v6-inv_model.pth")['optimizer'])

        self.transitions = self.gen_env_samples(self.env, policy=None, num_samples=self.buffer_size, env_type='continuous', use_policy=False)
        self.train_env(self.transitions)
        # self.train_inv(self.transitions)
        self.step_num = 0
            
    def forward(self, states, actions):
        '''
        timesteps: bs * (context_len+1)
        states: bs * (context_len+1) * state_dim
        actions: bs * (context_len+1) * act_dim
        '''
        energy = self.energy_model(states, actions)
        return energy

    def configure_optimizers(self):
        # self.optimizer_energy = optim.Adam(self.energy_model.parameters(), lr=self.lr)
        self.optimizer_energy = optim.AdamW(self.energy_model.parameters(), lr=self.lr)
        # self.scheduler_energy = optim.lr_scheduler.StepLR(self.optimizer_energy, step_size=100, gamma=0.98)
        # return [self.optimizer_energy], [self.scheduler_energy]
        return [self.optimizer_energy]

    def training_step(self, batch, batch_idx):
        '''
        timesteps: bs * (context_len+1)
        states: bs * (context_len+1) * state_dim
        actions: bs * (context_len+1) * act_dim
        '''

        timesteps, states, actions = batch
        pre_states, cur_states, next_states = states[:, :-1, :], states[:,-2:-1, :], states[:, -1:, :]
        # inv_actions = self.inv_model(cur_states.squeeze(1), next_states.squeeze(1)).reshape(self.batch_size, 1, self.act_dim)
        # inv_energy = self.energy_model(pre_states, inv_actions)
        # inv_loss = self.l1_loss(inv_actions, actions[:,-2:-1,:])
        # self.log('action/inv_gt_loss', inv_loss)

        timesteps, states, actions = batch
        # states = states.unsqueeze(1).repeat(1, self.n_prior_sample * self.select, 1, 1).reshape(self.batch_size * self.n_prior_sample * self.select, self.context_len+1, self.state_dim)
        states = states.unsqueeze(1).repeat(1, self.n_prior_sample, 1, 1).reshape(self.batch_size * self.n_prior_sample, self.context_len+1, self.state_dim)
        pre_states, cur_states, next_states = states[:, :-1, :], states[:,-2:-1, :], states[:, -1:, :]
        
        prior_actions = prior_sample_new_exmps(self.energy_model,
                                               self.n_prior_sample * self.batch_size, 
                                               pre_states, 
                                               steps=self.prior_sampler_steps, 
                                               act_dim=self.act_dim,
                                               stepsize_init=self.prior_sampler_stepsize_init,
                                               stepsize_end=self.prior_sampler_stepsize_end,
                                               init_act=None,
                                               add_noise=True, 
                                               noise_scale=self.noise_scale,
                                               optimal=False)

        # prior_actions = prior_sample_new_exmps(self.energy_model,
        #                                        self.select * self.n_prior_sample * self.batch_size, 
        #                                        pre_states, 
        #                                        steps=self.prior_sampler_steps, 
        #                                        act_dim=self.act_dim,
        #                                        stepsize_init=self.prior_sampler_stepsize_init,
        #                                        stepsize_end=self.prior_sampler_stepsize_end,
        #                                        init_act=None,
        #                                        add_noise=True, 
        #                                        noise_scale=self.noise_scale,
        #                                        optimal=False)
        # pred_1 = self.env_model_1(pre_states.squeeze(1), prior_actions.squeeze(1))
        # pred_2 = self.env_model_2(pre_states.squeeze(1), prior_actions.squeeze(1))
        # discrepancy = torch.abs(pred_1 - pred_2).sum(dim=-1).reshape(self.batch_size, self.n_prior_sample * self.select)
        # sorted_logits, sorted_indices = torch.sort(discrepancy, dim=-1)
        # prior_actions = prior_actions.reshape(self.batch_size, self.n_prior_sample * self.select, self.act_dim)
        # prior_actions = torch.gather(prior_actions, 1, sorted_indices[:,:self.n_prior_sample].unsqueeze(-1).expand(-1, -1, self.act_dim)).reshape(self.batch_size * self.n_prior_sample, 1, self.act_dim)

        timesteps, states, actions = batch
        states = states.unsqueeze(1).repeat(1, self.n_prior_sample, 1, 1).reshape(self.batch_size * self.n_prior_sample, self.context_len+1, self.state_dim)
        pre_states, cur_states, next_states = states[:, :-1, :], states[:,-2:-1, :], states[:, -1:, :]
        
        

        prior_actions.requires_grad = True
        energy = self.energy_model(pre_states, prior_actions).reshape(self.batch_size, self.n_prior_sample)

        # idx = (energy < inv_energy.repeat(1, self.n_prior_sample)).reshape(self.batch_size * self.n_prior_sample,)
        # prior_actions = prior_actions.squeeze(1)
        # prior_actions[idx] = inv_actions.repeat(1, self.n_prior_sample, 1).reshape(self.batch_size * self.n_prior_sample, self.act_dim)[idx]
        # prior_actions = prior_actions.unsqueeze(1)
        # prior_actions = prior_actions.detach()

        final_grad = torch.autograd.grad(outputs=energy.sum(), inputs=prior_actions, retain_graph=True, create_graph=True, only_inputs=True)[0]
        norm = torch.norm(final_grad, p=float('inf'), dim=-1)
        grad_loss = torch.mean(torch.clamp(norm - 2., min=0.) ** 2)

        self.env_pdf = MultivariateNormal(loc=torch.zeros(self.state_dim).to('cuda'), covariance_matrix=torch.eye(self.state_dim).to('cuda')*self.sigma)
        env_error = next_states.squeeze(1) - self.env_model_1(pre_states.squeeze(1), prior_actions.squeeze(1))
        log_probs = self.env_pdf.log_prob(env_error)
        thr = - self.state_dim / 2 * math.log(2 * math.pi) - self.state_dim * math.log(self.sigma) - self.state_dim / 2
        # log_probs[log_probs<-20] = -20
        log_probs[log_probs<thr] = thr
        env_probs = log_probs.exp().reshape(self.batch_size, self.n_prior_sample)
        weight = env_probs.mean(dim=1, keepdim=True) ** (-1) * env_probs - 1.


        weighted_ebm_loss = -(energy * weight).mean()
        weighted_reg_loss = (energy ** 2).mean()

        prior_actions = prior_actions.reshape(self.batch_size, self.n_prior_sample, self.act_dim)

        self.log('action/ebm_gt_loss', self.l1_loss(prior_actions, actions[:,-2:-1,:].repeat(1, self.n_prior_sample, 1)))
        # self.log('action/inv_ebm_loss', self.l1_loss(prior_actions, inv_actions.repeat(1, self.n_prior_sample, 1)))

        # targets = torch.cat([actions[:,-2:-1,:], prior_actions], dim=1).reshape(self.batch_size*(self.n_prior_sample+1), -1, self.act_dim).detach()
        # targets = torch.cat([inv_actions, prior_actions], dim=1).reshape(self.batch_size*(self.n_prior_sample+1), -1, self.act_dim).detach()
        # targets.requires_grad = True
        # timesteps, states, actions = batch
        # states = states.unsqueeze(1).repeat(1, self.n_prior_sample+1, 1, 1).reshape(self.batch_size * (self.n_prior_sample+1), self.context_len+1, self.state_dim)
        # pre_states, cur_states, next_states = states[:, :-1, :], states[:,-2:-1, :], states[:, -1:, :]
        # energy = self.energy_model(pre_states, targets).reshape(self.batch_size, self.n_prior_sample+1)
        # reg_loss = (energy ** 2).mean()

        # pos_energy = energy[:,0].mean()
        # neg_energy = energy[:,0:].mean()

        # ebm_loss = neg_energy - pos_energy

        # softmax_energy = F.softmax(energy, dim=-1)
        # labels = torch.zeros((self.batch_size, self.n_prior_sample+1)).to('cuda')
        # labels[:,0] = 1.
        # ebm_loss = nn.KLDivLoss(reduction='batchmean')(labels, softmax_energy)

        # total_loss = ebm_loss 
        # total_loss = ebm_loss + reg_loss
        # total_loss = ebm_loss + grad_loss
        # total_loss = ebm_loss + weighted_ebm_loss + grad_loss
        total_loss = weighted_ebm_loss + weighted_reg_loss * 0.0001 + grad_loss * 0.001
        # total_loss = weighted_ebm_loss + grad_loss + reg_loss * 0.0001
        # total_loss = weighted_ebm_loss + grad_loss
        
        # total_loss = weighted_ebm_loss + weighted_reg_loss * 0.0001 if self.current_epoch <= self.num_epoch_train_env else ebm_loss + reg_loss * 0.01
        # total_loss = weighted_ebm_loss + weighted_reg_loss * 0.00001 + ebm_loss + reg_loss * 0.01


        self.log_dict({
                'train/sigma': self.sigma,
                'train/two_sigma_rate': (log_probs>=thr).sum() / len(log_probs),
                # 'train/grad_loss': grad_loss,
                # 'train/pos_energy': pos_energy,
                # 'train/neg_energy': neg_energy,
                'train/total_loss': total_loss,
                # 'train/reg_loss': reg_loss,
                # 'train/final_grad': norm.mean(),
                'train/weighted_ebm_loss': weighted_ebm_loss,
                'train/weighted_reg_loss': weighted_reg_loss,
                # 'train/ebm_loss': ebm_loss,
                },
                prog_bar=True, logger=True)

        return total_loss
        return ebm_loss

    def training_epoch_end(self, outputs):
        if self.current_epoch % self.num_epoch_collect != self.num_epoch_collect - 1:
            return
        
        self.eval()
        rewards, lengths, energies, transitions = self.evaluate(self.num_eval_episodes)
        self.step_num += lengths.sum()
        self.transitions += transitions
        self.transitions = self.transitions[-self.buffer_size:]
        self.train()
        
        self.log_dict({
            'val/step_num': self.step_num,
            'val/val_reward_mean': np.mean(rewards),
            'val/val_reward_std': np.std(rewards),
            'val/val_lengths_mean': np.mean(lengths),
            'val/val_lengths_std': np.std(lengths)},
            prog_bar=False, logger=True)
        
        if self.current_epoch <= self.num_epoch_train_env:
            self.train_env(self.transitions)
            # self.train_inv(self.transitions)
            # if self.current_epoch <= 5000:
            self.sigma -= (0.01 - 0.0001) / (self.num_epoch_train_env / self.num_epoch_collect)
        return 
    
    def collect_data(self, num_eval_episodes):
        dones = np.bool8(np.zeros(self.num_eval_episodes))
        transitions = []

        obs = torch.from_numpy(test_env.reset()).to('cuda')
        states = torch.zeros((self.num_eval_episodes, self.max_test_ep_len+self.context_len-1, self.state_dim), device=obs.device)
        running_state = obs
        update_states = [running_state[i].cpu() for i in range(len(running_state))]

        for t in range(self.context_len, self.context_len + self.max_test_ep_len):
            if dones.sum() == self.num_eval_episodes:
                break
            
            states[:, t-1] = running_state

            act = test_env.action_space.sample()

            obs, reward, done, info = test_env.step(act)
            running_state = running_state.detach().cpu().numpy()

            for i in range(len(done)):
                if done[i] == 0:
                    update_states.append(torch.from_numpy(running_state[i]))
                    # transitions.append([running_state[i], act[i], obs[i]])
                    transitions.append(np.concatenate([running_state[i], act[i], obs[i]]))

            dones |= np.bool8(done)

            running_state = torch.from_numpy(obs)
        return np.stack(transitions)

    def evaluate(self, num_eval_episodes):
        dones = np.bool8(np.zeros(self.num_eval_episodes))
        episode_energies = np.zeros(self.num_eval_episodes)
        episode_rewards = np.zeros(self.num_eval_episodes)
        episode_lengths = np.zeros(self.num_eval_episodes)
        episode_transitions = [[] for _ in range(self.num_eval_episodes)]
        energies, transitions = [], []

        obs = torch.from_numpy(test_env.reset()).to('cuda')
        timesteps = torch.arange(start=0, end=self.max_test_ep_len, step=1).repeat(self.num_eval_episodes, 1).to(obs.device)
        pretimesteps = torch.zeros((self.num_eval_episodes, self.context_len-1), dtype=torch.int32, device=obs.device)
        timesteps = torch.cat((pretimesteps, timesteps), dim=1)
        timesteps = torch.cat((pretimesteps, timesteps), dim=1)

        actions = torch.zeros((self.num_eval_episodes, self.max_test_ep_len+self.context_len-1, self.act_dim), device=obs.device)
        states = torch.zeros((self.num_eval_episodes, self.max_test_ep_len+self.context_len-1, self.state_dim), device=obs.device)
        running_state = obs
        update_states = [running_state[i].cpu() for i in range(len(running_state))]

        for t in range(self.context_len, self.context_len + self.max_test_ep_len):
            # print(dones.sum())
            # if t == self.context_len + self.max_test_ep_len - 1:
            #     print(t)
            if dones.sum() == self.num_eval_episodes:
                break
            
            states[:, t-1] = running_state
            inp_states = states[:,t-self.context_len:t].unsqueeze(1).repeat(1, self.n_prior_sample_val, 1, 1).reshape(self.num_eval_episodes * self.n_prior_sample_val, self.context_len, self.state_dim).detach()
            # act_preds = self.prior_sampler.prior_sample_new_exmps(self.num_eval_episodes, states[:,t-self.context_len:t], steps=self.prior_sampler_steps, add_noise=False, optimal=True)
            # act_preds = self.prior_sampler.prior_sample_new_exmps(self.num_eval_episodes * self.n_prior_sample_val, inp_states, steps=self.prior_sampler_steps, add_noise=False, optimal=True, train=False)
            
            act_preds = prior_sample_new_exmps(self.energy_model,
                                               self.num_eval_episodes * self.n_prior_sample_val, 
                                               inp_states,
                                               steps=self.prior_sampler_steps, 
                                               act_dim=self.act_dim,
                                               stepsize_init=self.prior_sampler_stepsize_init,
                                               stepsize_end=self.prior_sampler_stepsize_end,
                                               init_act=None,
                                               add_noise=True, 
                                               noise_scale=self.noise_scale,
                                               optimal=False)
            act_preds = prior_sample_new_exmps(self.energy_model,
                                               self.num_eval_episodes * self.n_prior_sample_val, 
                                               inp_states,
                                               steps=self.prior_sampler_steps, 
                                               act_dim=self.act_dim,
                                               stepsize_init=self.prior_sampler_stepsize_end,
                                               stepsize_end=self.prior_sampler_stepsize_end,
                                               init_act=None,
                                               add_noise=True, 
                                               noise_scale=self.noise_scale,
                                               optimal=False)

            probs = get_probabilities(self.energy_model,
                                      self.num_eval_episodes,
                                      self.n_prior_sample_val,
                                      inp_states,
                                      act_preds)

            act = act_preds[:, -1].reshape(self.num_eval_episodes, self.n_prior_sample_val, self.act_dim).detach()

            idx = Categorical(probs).mode
            for i in range(self.num_eval_episodes):
                actions[i, t-1] = act[i][idx[i]]

            
            act = actions[:, t-1]
            act += torch.randn_like(act) * 0.1
            act = act.clamp(-1., 1.)
            act = act.detach().cpu().numpy()

            obs, reward, done, info = test_env.step(act)
            running_state = running_state.detach().cpu().numpy()

            for i in range(len(done)):
                if done[i] == 0:
                    update_states.append(torch.from_numpy(running_state[i]))
                    episode_transitions[i].append([running_state[i], act[i], obs[i]])
                    # energies.append(eeee[i])
                    transitions.append([running_state[i], act[i], obs[i]])
                    # transitions.append(np.hstack([running_state[i], act[i], obs[i]]))

            # episode_energies[~dones] += eeee[~dones]
            episode_rewards[~dones] += reward[~dones]
            episode_lengths[~dones] += 1
            dones |= np.bool8(done)
            # print(dones.sum())

            running_state = torch.from_numpy(obs)
        # return episode_rewards, episode_lengths, episode_energies, episode_transitions
        # return episode_rewards, episode_lengths, energies, np.vstack(transitions)
        return episode_rewards, episode_lengths, energies, transitions

    def gen_env_samples(self, env, policy, num_samples, env_type, use_policy):
        count = 0
        transitions = []
        s = env.reset()
        t = 0
        while count < num_samples:
            if env_type == 'continuous':
                if use_policy:
                    a = select_action_continuous(s, policy)
                else:
                    a = env.action_space.sample()
            else:
                a = select_action_discrete(s, policy)
            s_prime, _, done, _ = env.step(a)
            transitions.append([s, a, s_prime])
            # transitions.append(np.hstack((s, a, s_prime)))
            count += 1
            t += 1
            if done == True or t>self.max_test_ep_len:
                s = env.reset()
                t = 0
            else:
                s = s_prime
        self.transitions += transitions
        # return np.vstack(transitions)
        return transitions

    def init_env(self, dataset):
        # state = rawdata['state'].float().cuda()
        # action = rawdata['action'].float().cuda() 
        # next_state = rawdata['next_state'].float().cuda()
        s = dataset['state'].cuda()
        a = dataset['action'].cuda()
        ns = dataset['next_state'].cuda()
        # delta = dataset['delta'].cuda()
        # s_norm = dataset['state_norm'].cuda()
        # a_norm = dataset['action_norm'].cuda()
        # ns_norm = dataset['next_state_norm'].cuda()
        # delta_norm = dataset['delta_norm'].cuda()
        # s_mean = dataset['state_mean'].cuda()
        # a_mean = dataset['action_mean'].cuda()
        # ns_mean = dataset['next_state_mean'].cuda()
        # delta_mean = dataset['delta_mean'].cuda()
        # s_std = dataset['state_std'].cuda()
        # a_std = dataset['action_std'].cuda()
        # ns_std = dataset['next_state_std'].cuda()
        # delta_std = dataset['delta_std'].cuda()
        # state = rawdata['state'].float().cuda()
        # action = rawdata['action'].float().cuda()
        # next_state = rawdata['next_state'].float().cuda()
        # s = torch.cat((s, state), dim=0)
        # a = torch.cat((a, action), dim=0)
        # ns = torch.cat((ns, next_state), dim=0)
        min_loss = 1000000.
        # transitions = []
        # d = torch.load('exp2/data/hopper_demos-3000.pth')
        # s, a, ns = d['state'].numpy(), d['action'].numpy(), d['next_state'].numpy()
        # for i in range(len(s)):
        #     transitions.append([s[i], a[i], ns[i]])

        loader = DataLoader(list(zip(s, a, ns)), shuffle=True, batch_size=150000)
        # loader = DataLoader(list(zip(state, action, next_state)), shuffle=True, batch_size=150000)

        min_loss = 100.
        epochs = 10000
        for epoch in range(epochs): 
            mse_loss = []
            running_loss = []
            for s, a, ns in loader:
                self.env_opt.zero_grad()
                s_pred = self.env_model(s.float(), a.float())
                # delta_pred = self.env_model(s.float(), a.float())
                loss = nn.MSELoss()(s_pred, ns.float())
                # loss = nn.MSELoss()(delta_pred, delta.float())
                loss.backward()
                self.env_opt.step()
                running_loss.append(nn.L1Loss()(s_pred, ns.float()).detach().cpu().numpy())
                # running_loss.append(nn.L1Loss()(delta_pred, delta.float()).detach().cpu().numpy())
                mse_loss.append(loss.detach().cpu().numpy())
            
            # state = rawdata['state'].float().cuda()
            # action = rawdata['action'].float().cuda() 
            # next_state = rawdata['next_state'].float().cuda()
            # test_loss = nn.L1Loss()(self.env_model(state, action), next_state)
            
            
            # self.log('train/env_loss', test_loss)
            # print('epoch: ', epoch, ' test loss: ', test_loss, ' train loss: ', np.mean(running_loss), np.mean(mse_loss))
            print('epoch: ', epoch, ' train loss: ', np.mean(running_loss), np.mean(mse_loss))

    def train_env(self, transitions):
        min_loss = 1000000.
        # transitions = []
        # d = torch.load(self.data_file)
        # s, a, ns = d['state'].numpy(), d['action'].numpy(), d['next_state'].numpy()
        # for i in range(len(s)):
        #     transitions.append([s[i], a[i], ns[i]])

        print('Learning environment model....')
        env_dataset_list = []
        env_dataset = transition_dataset(transitions)
        env_dataset_list.append(env_dataset)
        env_dataset_final = ConcatDataset(env_dataset_list)
        env_loader = DataLoader(env_dataset_final, batch_size=32768, shuffle=True, num_workers=0)
        
        epochs = 200 if self.current_epoch == 0 else 50
        for epoch in range(epochs): 
            train_loss_1, train_loss_2 = [], []
            for i, data in enumerate(env_loader):
                s, a, s_prime = data

                self.env_opt_1.zero_grad()
                s_pred_1 = self.env_model_1(s.float().cuda(), a.float().cuda())
                loss_1 = self.mse_loss(s_pred_1, s_prime.float().cuda())
                loss_1.backward()
                self.env_opt_1.step()
                train_loss_1.append(loss_1.detach().cpu().numpy())

                # self.env_opt_2.zero_grad()
                # s_pred_2 = self.env_model_2(s.float().cuda(), a.float().cuda())
                # loss_2 = self.mse_loss(s_pred_2, s_prime.float().cuda())
                # loss_2.backward()
                # self.env_opt_2.step()
                # train_loss_2.append(loss_2.detach().cpu().numpy())

            data = torch.load(self.data_file)
            state = data['state'].float().cuda()
            action = data['action'].float().cuda()
            next_state = data['next_state'].float().cuda()
            test_loss_1 = nn.L1Loss()(self.env_model_1(state, action), next_state)
            # test_loss_2 = nn.L1Loss()(self.env_model_2(state, action), next_state)
            # train_loss_1 = np.mean(train_loss_1)
            # train_loss_2 = np.mean(train_loss_2)
            self.log('env/env_test_loss_1', test_loss_1)
            # self.log('env/env_test_loss_2', test_loss_2)
            # self.log('env/env_train_loss_1', train_loss_1)
            # self.log('env/env_train_loss_2', train_loss_2)
            print('epoch: ', epoch, 'env_loss: ', np.mean(train_loss_1), test_loss_1.item())
        state_dict_1 = {"net": self.env_model_1.state_dict(), "optimizer": self.env_opt_1.state_dict(), "epoch": epoch}
        torch.save(state_dict_1, "exp2/weight/Hopper-v6-env_model_1.pth")
        # state_dict_2 = {"net": self.env_model_2.state_dict(), "optimizer": self.env_opt_2.state_dict(), "epoch": epoch}
        # torch.save(state_dict_2, "exp2/weight/Hopper-v6-env_model_2.pth")
            # if test_loss_1.item() <= 0.002:
            #     exit()

    
    def train_inv(self, transitions):
        min_loss = 1000000.
        # transitions = []
        # d = torch.load(self.data_file)
        # s, a, ns = d['state'].numpy(), d['action'].numpy(), d['next_state'].numpy()
        # for i in range(len(s)):
        #     transitions.append([s[i], a[i], ns[i]])

        print('Learning inverse model....')
        inv_dataset_list = []
        inv_dataset = transition_dataset(transitions)
        inv_dataset_list.append(inv_dataset)
        inv_dataset_final = ConcatDataset(inv_dataset_list)
        inv_loader = DataLoader(inv_dataset_final, batch_size=32768, shuffle=True, num_workers=0)\
        
        epochs = 200 if self.current_epoch == 0 else 50
        for epoch in range(epochs): 
            train_loss = []
            for i, data in enumerate(inv_loader):
                s, a, s_prime = data
                self.inv_opt.zero_grad()
                a_pred = self.inv_model(s.float().cuda(), s_prime.float().cuda())
                loss = self.mse_loss(a_pred, a.float().cuda())
                loss.backward()
                train_loss.append(loss.detach().cpu().numpy())
                self.inv_opt.step()
                        
            data = torch.load(self.data_file)
            state = data['state'].float().cuda()
            action = data['action'].float().cuda()
            next_state = data['next_state'].float().cuda()
            test_loss = nn.L1Loss()(self.inv_model(state, next_state), action)
            self.log('env/inv_test_loss', test_loss)
            # self.log('env/inv_train_loss', train_loss)
            # print('epoch: ', epoch, 'inv_loss: ', np.mean(train_loss), test_loss.item())
        state_dict = {"net": self.inv_model.state_dict(), "optimizer": self.inv_opt.state_dict(), "epoch": epoch}
        torch.save(state_dict, "exp2/weight/Hopper-v6-inv_model.pth")


    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(
                parents=[parent_parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--env_name', type=str, default="Hopper-v6")
        parser.add_argument('--data_file', type=str, default="exp2/data/hopper_demos_sac700.pth")
        parser.add_argument('--lr', type=float, default=5e-3)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--h_dim', type=int, default=64)
        parser.add_argument('--depth', type=int, default=8)
        parser.add_argument('--context_len', type=int, default=1)
        parser.add_argument('--n_prior_sample', type=int, default=8)
        parser.add_argument('--n_prior_sample_val', type=int, default=512)
        parser.add_argument('--prior_sampler_steps', type=int, default=40)
        parser.add_argument('--prior_sampler_stepsize_init', type=float, default=10)
        parser.add_argument('--prior_sampler_stepsize_end', type=float, default=1)
        parser.add_argument('--noise_scale', type=float, default=0.1)
        parser.add_argument('--regularization_alpha', type=float, default=0.01)
        parser.add_argument('--max_test_ep_len', type=int, default=400)
        parser.add_argument('--sigma', type=float, default=0.1)
        parser.add_argument('--num_eval_episodes', type=int, default=50)
        parser.add_argument('--num_epoch_collect', type=int, default=20)
        parser.add_argument('--num_epoch_eval', type=int, default=100)
        parser.add_argument('--num_epoch_train_env', type=int, default=5000)
        parser.add_argument('--error_bound', type=float, default=0.001)
        parser.add_argument('--buffer_size', type=int, default=100000)
        parser.add_argument('--select', type=int, default=2)
        return parser





model_parser = DEBM.add_model_specific_args()
model_args, _ = model_parser.parse_known_args()
trainer_parser = argparse.ArgumentParser(add_help=False)
model_args = vars(model_args)
CHECKPOINT_PATH = "exp2/saved_models/" + model_args['env_name']
trainer_parser.add_argument('--max_epochs', type=int, default=100000)
trainer_parser.add_argument('--gradient_clip_val', type=float, default=0.5)
trainer_parser.add_argument('--default_root_dir', type=str, default=CHECKPOINT_PATH)
trainer_parser.add_argument('--log_every_n_steps', type=int, default=1)

trainer_args, _ = trainer_parser.parse_known_args()
trainer_args = vars(trainer_args)

env = GymEnv(gym.make(model_args['env_name']))
test_env = gym.vector.make(model_args['env_name'], num_envs=model_args['num_eval_episodes'])
env.set_seed(seed)
test_env.seed(seed)
env.action_space.seed(seed)
test_env.action_space.seed(seed)

rawdata = torch.load(model_args['data_file'])
train_dataset = TrainingSet(rawdata, env, context_len=model_args['context_len'])
train_loader = data.DataLoader(train_dataset, batch_size=model_args['batch_size'], shuffle=False, drop_last=True, num_workers=0, pin_memory=False)

data_states = rawdata['state']
data_next_states = rawdata['next_state']

# try:
#     version_id = max([int(l.split('_')[1]) for l in os.listdir(trainer_args['default_root_dir'] + '/lightning_logs')]) + 1
# except:
#     version_id = 0
version_id = 72
# print('version: ', version_id)

# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#                 monitor='train/total_loss',
#                 # dirpath=trainer_args['default_root_dir'] + 'checkpoints/version_' + str(version_id),
#                 # filename='exp-version=' + str(version_id) + '-{epoch:02d}.pl',
#                 save_top_k=-1,
#                 every_n_epochs=500,
#                 mode='min')

name = '-'.join([model_args['env_name'], str(model_args['lr']), str(model_args['h_dim']), str(model_args['n_prior_sample']), str(model_args['num_eval_episodes']), str(model_args['num_epoch_collect']), str(model_args['buffer_size'])])
wandb_logger = WandbLogger(project='VEBM', name=name, log_model=True, save_dir=trainer_args['default_root_dir'] + '/checkpoints/version_' + str(version_id))

model = DEBM(env, 0, **model_args)
# trainer = pl.Trainer(**trainer_args, accelerator='gpu', devices=1, callbacks=[checkpoint_callback], logger=wandb_logger)
trainer = pl.Trainer(**trainer_args, accelerator='gpu', devices=1, logger=wandb_logger)
trainer.fit(model, train_loader)


