import os
import io
import argparse
import numpy as np
import random
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from torch.distributions.multivariate_normal import MultivariateNormal
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

sns.reset_orig()
matplotlib.rcParams['lines.linewidth'] = 2.0
CHECKPOINT_PATH = "exp/saved_models/cubic_final/"
pl.seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
action_delta = torch.ones(1, device='cuda') * 0.5


def real_env(states, actions, std=None):
    if std:
        noise = torch.randn(actions.shape, device=actions.device)
        noise.normal_(0, std)
    else:
        noise = 0.
    states[:,0:1] += actions + noise
    states[:,1:2] += 0.1
    return states

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EnergyFunc(nn.Module):
    def __init__(self, state_dim=2, act_dim=2, h_dim=8, context_len=5):
        super(EnergyFunc, self).__init__()
        self.energyfunc = nn.Sequential(
            nn.Linear(context_len * state_dim + act_dim, context_len * h_dim),
            Swish(),
            nn.Linear(context_len * h_dim, context_len * h_dim),
            Swish(),
            nn.Linear(context_len * h_dim, 1)
        )
    
    def forward(self, states, actions):
        x = torch.cat((states.reshape(states.shape[0], -1), actions.squeeze(1)), dim=-1)
        return self.energyfunc(x)

class PriorSampler:
    def __init__(self, energy_model, act_dim, context_len, max_test_ep_len):
        super(PriorSampler, self).__init__()

        self.energy_model = energy_model
        self.act_dim = act_dim
        self.context_len = context_len
        self.max_test_ep_len = max_test_ep_len
    
    def prior_sample_planning_act_first(self, states, actions, goals, start_time, steps, step_size, planning_target_alpha, add_noise=False, optimal=False):
        is_training = self.energy_model.training
        self.energy_model.eval()
        for p in self.energy_model.parameters():
            p.requires_grad = False
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        noise = torch.randn(actions.shape, device=actions.device)

        for _ in range(steps):
            actions = actions.detach()
            actions.requires_grad = True
            states = states.detach()

            if add_noise:
                noise.normal_(0, 0.02)
                actions.data.add_(noise.data)
                actions[:,:,0].data.clamp_(min=-action_delta, max=action_delta)            

            energy_loss = 0.
            gamma = 0.8
            for t in range(start_time, self.context_len + self.max_test_ep_len - 1):
                # energy_loss += -self.energy_model(states[:,t-self.context_len:t], actions[:,t-1:t]).sum() * gamma ** (t - start_time)
                energy_loss += -self.energy_model(states[:,t-self.context_len:t], actions[:,t-1:t]).sum()
                states[:, t, 0] = states[:, t-1, 0] +  actions[:, t-1, 0]
                # states.detach_()
            target_loss = torch.nn.MSELoss()(goals, states[:, -1, 0:1])
            # target_loss = torch.nn.HuberLoss(delta=0.2)(goals, states[:, -1, 0:1])
            # target_loss = torch.nn.HuberLoss(delta=0.2)(goals, states[:, -1, 0:1]) * ((start_time -6-10) / (25-10-6)) ** 3 * 2
            planning_loss = energy_loss + target_loss * planning_target_alpha
            # planning_loss = energy_loss
            print('energy_loss: ', energy_loss.item(), ' target_loss: ', torch.nn.L1Loss()(goals, states[:, -1, 0:1]).item())

            planning_loss.backward()
            actions.grad.data.clamp_(-0.3, 0.3)

            actions.data.add_(-step_size * actions.grad.data)
            actions.grad.detach_()
            actions.grad.zero_()
            actions[:,:,0].data.clamp_(min=-action_delta, max=action_delta)

        for p in self.energy_model.parameters():
            p.requires_grad = True
        self.energy_model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)
        
        return actions[:,start_time-1:start_time,:]

    def prior_sample_planning(self, states, actions, goals, steps, step_size, planning_target_alpha, add_noise=False, optimal=False):
        is_training = self.energy_model.training
        self.energy_model.eval()
        for p in self.energy_model.parameters():
            p.requires_grad = False
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        noise = torch.randn(actions.shape, device=actions.device)

        for _ in range(steps):
            actions = actions.detach()
            actions.requires_grad = True
            states = states.detach()

            if add_noise:
                noise.normal_(0, 0.02)
                actions.data.add_(noise.data)
                actions[:,:,0].data.clamp_(min=-action_delta, max=action_delta)            

            energy_loss = 0.
            for t in range(self.context_len, self.context_len + self.max_test_ep_len - 1):
                energy_loss += -self.energy_model(states[:,t-self.context_len:t], actions[:,t-1:t]).sum()
                states[:, t, 0] = states[:, t-1, 0] +  actions[:, t-1, 0]
            target_loss = torch.nn.MSELoss()(goals, states[:, -1, 0:1])
            planning_loss = energy_loss + target_loss * planning_target_alpha
            print('energy_loss: ', energy_loss.item(), ' target_loss: ', torch.nn.L1Loss()(goals, states[:, -1, 0:1]).item())

            planning_loss.backward()
            actions.grad.data.clamp_(-0.3, 0.3)

            actions.data.add_(-step_size * actions.grad.data)
            actions.grad.detach_()
            actions.grad.zero_()
            actions[:,:,0].data.clamp_(min=-action_delta, max=action_delta)

        for p in self.energy_model.parameters():
            p.requires_grad = True
        self.energy_model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)
        
        return actions, states

    def prior_sample_new_exmps(self, states, steps, step_size, add_noise=False, optimal=False):
        is_training = self.energy_model.training
        self.energy_model.eval()
        for p in self.energy_model.parameters():
            p.requires_grad = False
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        inp_actions = torch.rand((states.shape[0], 1, self.act_dim), device=states.device) * 2 * action_delta - action_delta
        inp_actions.requires_grad = True

        noise = torch.randn(inp_actions.shape, device=inp_actions.device)

        energy_min = torch.ones(states.shape[0], device=inp_actions.device) * np.inf
        action_min = torch.zeros_like(inp_actions)
        for _ in range(steps):
            if add_noise:
                noise.normal_(0, 0.02)
                inp_actions.data.add_(noise.data)
                inp_actions[:,:,0].data.clamp_(min=-action_delta, max=action_delta)
            
            energy = -self.energy_model(states, inp_actions).squeeze(-1)
            
            action_min[energy<energy_min]=inp_actions[energy<energy_min]
            energy_min[energy<energy_min]=energy[energy<energy_min]
            
            energy.sum().backward()
            inp_actions.grad.data.clamp_(-0.3, 0.3)

            inp_actions.data.add_(-step_size * inp_actions.grad.data)
            inp_actions.grad.detach_()
            inp_actions.grad.zero_()
            inp_actions[:,:,0].data.clamp_(min=-action_delta, max=action_delta)

        for p in self.energy_model.parameters():
            p.requires_grad = True
        self.energy_model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)
        
        if optimal:
            return action_min
        else:
            return inp_actions

class DEBM(pl.LightningModule):
    def __init__(self, 
        env_training_dataloader=None, 
        lr=3e-4, 
        batch_size=256, 
        state_dim=2, 
        act_dim=2,
        h_dim=8, 
        context_len=10, 
        n_prior_sample=1,
        prior_sampler_steps=5,
        prior_sampler_stepsize=5,
        prior_sampler_steps_val=5,
        prior_sampler_stepsize_val=5,
        prior_sampler_steps_plan=5,
        prior_sampler_stepsize_plan=5,
        regularization_alpha = 1.,
        max_test_ep_len=100,
        sigma=0.1,
        planning_target_alpha=10.):
        super(DEBM, self).__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.n_prior_sample = n_prior_sample
        self.prior_sampler_steps=prior_sampler_steps
        self.prior_sampler_stepsize=prior_sampler_stepsize
        self.prior_sampler_steps_val=prior_sampler_steps_val
        self.prior_sampler_stepsize_val=prior_sampler_stepsize_val
        self.prior_sampler_steps_plan=prior_sampler_steps_plan
        self.prior_sampler_stepsize_plan=prior_sampler_stepsize_plan
        self.regularization_alpha = regularization_alpha
        self.max_test_ep_len = max_test_ep_len
        self.env_sigma = sigma
        self.planning_target_alpha = planning_target_alpha

        self.energy_model = EnergyFunc(state_dim=state_dim, act_dim=act_dim, h_dim=h_dim, context_len=context_len)
        self.env_model = nn.Linear(1 + act_dim, 1, bias=False)
        self.env_model.load_state_dict(torch.load('exp/weight/env_params.pth'))
        self.env_pdf = MultivariateNormal(loc=torch.zeros(self.state_dim-1).to('cuda'), covariance_matrix=torch.eye(self.state_dim-1).to('cuda')*0.1)
        self.mse_loss = torch.nn.MSELoss()
        self.prior_sampler = PriorSampler(self.energy_model, act_dim, context_len, max_test_ep_len)
        
    def forward(self, states, actions):
        '''
        timesteps: bs * (context_len+1)
        states: bs * (context_len+1) * state_dim
        actions: bs * (context_len+1) * act_dim
        '''
        energy = self.energy_model(states, actions)
        return energy

    def configure_optimizers(self):
        optimizer_energy = optim.Adam(self.energy_model.parameters(), lr=self.lr)
        scheduler_energy = optim.lr_scheduler.StepLR(optimizer_energy, 1, gamma=0.999)
        # return [optimizer_energy], [scheduler_energy]
        return [optimizer_energy]

    def training_step(self, batch, batch_idx):
        '''
        timesteps: bs * (context_len+1)
        states: bs * (context_len+1) * state_dim
        actions: bs * (context_len+1) * act_dim
        '''
        # return 
        
        timesteps, states, actions = batch
        states = states.unsqueeze(1).repeat(1, self.n_prior_sample, 1, 1).reshape(self.batch_size * self.n_prior_sample, self.context_len+1, self.state_dim)
        pre_states, cur_states, next_states = states[:, :-1, :], states[:,-2:-1, :], states[:, -1:, :]
        
        prior_actions = self.prior_sampler.prior_sample_new_exmps(states[:, :-1], steps=self.prior_sampler_steps, 
                    step_size=self.prior_sampler_stepsize, add_noise=True, optimal=False).detach()
        
        env_error = next_states.squeeze(1)[:,0] - cur_states.squeeze(1)[:,0] - prior_actions.squeeze(-1).squeeze(-1)
        log_probs = self.env_pdf.log_prob(env_error.unsqueeze(1))
        log_probs[log_probs<-20] = -20
        env_probs = log_probs.exp().reshape(self.batch_size, self.n_prior_sample)
        weight = env_probs.mean(dim=1, keepdim=True) ** (-1) * env_probs - 1.

        prior_actions.requires_grad = True
        energy = self.energy_model(pre_states, prior_actions).reshape(self.batch_size, self.n_prior_sample)
        reg_loss = (energy ** 2).mean()
        
        weighted_ebm_loss = (energy * weight).mean()
        total_loss = -weighted_ebm_loss + reg_loss * self.regularization_alpha

        prior_actions = prior_actions.reshape(self.batch_size, self.n_prior_sample, self.act_dim)
        mse_loss = self.mse_loss(prior_actions, actions[:,-2:-1,:].repeat(1, self.n_prior_sample, 1))
        self.log('train/mse_loss', mse_loss)

        self.log_dict({
                'train/total_loss': total_loss,
                'train/weighted_ebm_loss': weighted_ebm_loss,
                },
                prog_bar=True, logger=True)
        return total_loss

    def plot_curve(self, states, linestyle, kind, goals=None):
        colors = ['red', 'blue', 'green', 'purple', 'black']
        assert len(colors) == states.shape[0]
        assert kind in ['normal', 'planning', 'once']
        
        figure = plt.figure()
        res_3 = []
        a_ = []
        max_y = []
        dp = []
        ddp = []
        x = [t for t in np.arange(-1.0, 1.1, .1)]
        for i in range(states.shape[0]):
            y = states[i, :, 0]
            p_fitted_3 = np.polynomial.Polynomial.fit(x, y, deg=3)
            dp_fitted_3 = p_fitted_3.deriv()
            ddp_fitted_3 = dp_fitted_3.deriv()
            residual_3 = np.abs(p_fitted_3(np.array(x)) - y).mean()
            a = np.abs(p_fitted_3.coef[-1])
            res_3.append(residual_3)
            a_.append(a)
            max_y.append(np.max(np.abs(y)))
            dp.append(np.mean(np.abs(dp_fitted_3(np.array(x)))))
            ddp.append(np.mean(np.abs(ddp_fitted_3(np.array(x)))))

            plt.plot(x, y, color=colors[i], linestyle=linestyle)
            if goals is not None:
                plt.scatter(1, goals[i,0].item(), marker='x', color=colors[i], linewidths=2)

        res_3 = np.array(res_3)
        max_y = np.array(max_y)
        a_ = np.array(a_)
        dp = np.array(dp)
        ddp = np.array(ddp)
        acc_id = a_ >= 0.5
        rej_num = len(a_) - np.sum(acc_id)
        acc_rate = np.sum(acc_id) / (np.sum(acc_id) + rej_num)
        acc_res_3 = np.mean(res_3[acc_id])
        self.log('val/acc_rate' + kind, acc_rate)
        self.log('val/acc_res_3' + kind, acc_res_3)
        self.logger.experiment.add_figure("val/traj" + kind, figure, global_step=self.global_step)

    def validation_step(self, batch, batch_idx):
        '''
        batch = start point: bs * state_dim
        '''
        if self.current_epoch % 10 != 0:
            return 

        test_num = 10
        batch = torch.rand((test_num, 1), device='cuda')*2-1
        batch = torch.cat((batch, torch.zeros_like(batch) - 1), dim=1)
        
        actions = torch.zeros((test_num, self.max_test_ep_len+self.context_len-1, self.act_dim), device=batch.device)
        states = torch.zeros((test_num, self.max_test_ep_len+self.context_len-1, self.state_dim), device=batch.device)
        running_state = batch
        for t in range(self.context_len, self.context_len + self.max_test_ep_len):
            states[:, t-1] = running_state
            act_preds = self.prior_sampler.prior_sample_new_exmps(states[:,t-self.context_len:t], steps=self.prior_sampler_steps_val,
                                                                   step_size=self.prior_sampler_stepsize_val, add_noise=True, optimal=False)
            act = act_preds[:, -1].detach()
            running_state = real_env(running_state, act, std=0.0)
            actions[:, t-1] = act
        
        self.plot_curve(states[:, self.context_len-1:t].detach().cpu().numpy(), linestyle='-', kind='normal')

        return

        if self.current_epoch < 1500:
            return 

        ########################################
        # goals = torch.rand((test_num,1), device='cuda')*2-1
        goals = (states[:,-1,0:1].clamp(-1,1) + torch.randn(5,1).to('cuda') * 0.2).clamp(-1,1)
        p_states = states.clone().detach()
        p_actions = actions.clone().detach()
        start_time = 10
        running_state = p_states[:, self.context_len-1+start_time].clone()
        for t in range(self.context_len + start_time, self.context_len + self.max_test_ep_len):
            p_states[:, t-1] = running_state.clone()
            if t == self.context_len + self.max_test_ep_len - 1:
                break
            act_preds = self.prior_sampler.prior_sample_planning_act_first(p_states, p_actions, goals, t,
                                    self.prior_sampler_steps_plan, self.prior_sampler_stepsize_plan, self.planning_target_alpha, add_noise=True, optimal=False)
            act = act_preds[:, -1].detach()
            running_state = real_env(running_state, act, std=0.0).clone()
            p_actions[:, t-1] = act

        self.plot_curve(p_states[:, self.context_len-1:t].detach().cpu().numpy(), linestyle=':', kind='once', goals=goals)
        np.save('exp/result_img/npy/'+str(self.current_epoch), p_states[:, self.context_len-1:t].detach().cpu().numpy())
        np.save('exp/result_img/npy/'+str(self.current_epoch)+'g', goals.detach().cpu().numpy())
        ########################################

        # goals = torch.rand((test_num,1), device='cuda')*2-1
        # planning_states = states.clone().detach()
        # planning_actions = actions.detach()

        # planning_actions, planning_states = self.prior_sampler.prior_sample_planning(planning_states, planning_actions, goals,
        #                         self.prior_sampler_steps_plan, self.prior_sampler_stepsize_plan, self.planning_target_alpha, add_noise=True, optimal=False)

        # # planning_states = planning_states[:, self.context_len-1:t].detach().cpu().numpy()
        # self.plot_curve(planning_states[:, self.context_len-1:t].detach().cpu().numpy(), linestyle=':', kind='planning', goals=goals)



    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(
                parents=[parent_parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--state_dim', type=int, default=2)
        parser.add_argument('--act_dim', type=int, default=1)
        parser.add_argument('--h_dim', type=int, default=4)
        parser.add_argument('--context_len', type=int, default=6)
        parser.add_argument('--n_prior_sample', type=int, default=4)
        parser.add_argument('--prior_sampler_steps', type=int, default=20)
        parser.add_argument('--prior_sampler_stepsize', type=float, default=1)
        parser.add_argument('--prior_sampler_steps_val', type=int, default=60)
        parser.add_argument('--prior_sampler_stepsize_val', type=float, default=1)
        parser.add_argument('--prior_sampler_steps_plan', type=int, default=500)
        parser.add_argument('--prior_sampler_stepsize_plan', type=float, default=0.5)
        parser.add_argument('--regularization_alpha', type=float, default=0.001)
        parser.add_argument('--max_test_ep_len', type=int, default=21)
        parser.add_argument('--sigma', type=float, default=0.1)
        parser.add_argument('--planning_target_alpha', type=float, default=1.)
        return parser

class TrainingSet(Dataset):
    def __init__(self, rawdata, context_len):
        super().__init__()
        # rawdata = rawdata.unsqueeze(-1)
        num, max_timestep, _ = rawdata.shape
        timesteps = torch.arange(start=0, end=max_timestep, step=1)
        timesteps = timesteps.repeat(num, 1).unsqueeze(-1)
        data_with_time = torch.cat((timesteps, rawdata), dim=-1)
        # done_time = torch.nonzero(data_with_time[:,:,-1])[:,1]
        # data_without_pad = []
        data_time = []
        data_state = []
        data_action = []
        for i in range(num):
            # print(i)
            data_without_pad = data_with_time[i, :, :]
            timesteps = data_without_pad[:, 0:1]
            states = data_without_pad[:, 1:]
            actions = data_without_pad[1:, 1:2] - data_without_pad[:-1, 1:2]
            length = actions.shape[0]
            for t in range(context_len):
                time = torch.zeros((1, context_len+1, 1), dtype=torch.int32)
                s = torch.zeros((1, context_len+1, states.shape[1]))
                a = torch.zeros((1, context_len+1, actions.shape[1]))
                time[0, -t-1:, :] = timesteps[:t+1, :].unsqueeze(0)
                s[0, -t-1:, :] = states[:t+1, :].unsqueeze(0)
                a[0, -t-1:, :] = actions[:t+1, :].unsqueeze(0)
                data_time.append(time)
                data_state.append(s)
                data_action.append(a)
            for t in range(length - context_len):
                data_time.append(timesteps[t:t+context_len+1, :].unsqueeze(0))
                data_state.append(states[t:t+context_len+1, :].unsqueeze(0))
                data_action.append(actions[t:t+context_len+1, :].unsqueeze(0))
        data_time = torch.vstack(data_time)
        data_state = torch.vstack(data_state)
        data_action = torch.vstack(data_action)
        data = torch.cat((data_time, data_state, data_action), dim=-1)
        idx = torch.randperm(data.shape[0])
        self.data = data[idx].view(data.size())
        print('training data loaded')
    
    def __getitem__(self, index):
        return self.data[index,:,0].to(torch.int32), self.data[index,:,1:3].to(torch.float32), self.data[index,:,3:].to(torch.float32)

    def __len__(self):
        return self.data.shape[0]
    
class ValidationSet(Dataset):
    def __init__(self, rawdata, eval_num=1000, from_training=True):
        super().__init__()
        
        self.data = None
        if from_training:
            start_point = []
            for i in range(rawdata.shape[0]):
                if not rawdata[i, 0, 1].item() in start_point:
                    start_point.append(rawdata[i, 0, 1].item())
            self.data = torch.vstack((-10*torch.ones(100), torch.tensor(start_point))).transpose(1,0)
        else:
            self.data = torch.rand((100,1))*2-1

        print('validation data loaded')
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def main():
    model_parser = DEBM.add_model_specific_args()
    model_args, ignored = model_parser.parse_known_args()
    model_args = vars(model_args)

    rawdata = torch.load('exp/data/cubic-10.pth')
    train_dataset = TrainingSet(rawdata, context_len=model_args['context_len'])
    val_dataset = ValidationSet(rawdata, eval_num=5, from_training=False)
    train_loader = data.DataLoader(train_dataset, batch_size=model_args['batch_size'], shuffle=True, drop_last=True, num_workers=8, pin_memory=False)
    val_loader = data.DataLoader(val_dataset, batch_size=100, shuffle=False, drop_last=True, num_workers=8, pin_memory=False)

    trainer_parser = argparse.ArgumentParser(add_help=False)
    trainer_parser.add_argument('--max_epochs', type=int, default=10000)
    trainer_parser.add_argument('--gradient_clip_val', type=float, default=0.5)
    trainer_parser.add_argument('--default_root_dir', type=str, default=CHECKPOINT_PATH)
    trainer_parser.add_argument('--log_every_n_steps', type=int, default=1)
    trainer_args, ignored = trainer_parser.parse_known_args()
    trainer_args = vars(trainer_args)

    try:
        version_id = max([int(l.split('_')[1]) for l in os.listdir(trainer_args['default_root_dir'] + '/lightning_logs')]) + 1
    except:
        version_id = 0
    print('version: ', version_id)


    model = DEBM(env_training_dataloader=None, **model_args)
    trainer = pl.Trainer(**trainer_args, accelerator='gpu', devices=1)
    trainer.fit(model, train_loader, val_loader, ckpt_path='exp/saved_models/cubic_final/lightning_logs/version_6/checkpoints/epoch=3832-step=11499.ckpt')
    # trainer.fit(model, train_loader, val_loader, ckpt_path='exp/saved_models/cubic_final/checkpoints/version_3/exp-version=3-epoch=1999.pl.ckpt')
    # trainer.fit(model, train_loader, val_loader, ckpt_path='exp/saved_models/cubic—planning/lightning_logs/version_20/checkpoints/epoch=3229-step=9690.ckpt')


if __name__ == '__main__':
    main()