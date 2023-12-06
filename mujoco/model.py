import torch
import torch.nn as nn
import torch.nn.functional as F

class dynamics_continuous(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(dynamics_continuous, self).__init__()
		self.linear_1 = nn.Linear(state_dim+action_dim, hidden_dim)
		self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_4 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_5 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_6 = nn.Linear(hidden_dim, state_dim)
		self.linear = [self.linear_1, self.linear_2, self.linear_3, self.linear_4, self.linear_5, self.linear_6]
		self._init_parameters()

	def forward(self, s, a):
		x = torch.cat([s, a], dim=1)
		x = self.linear_1(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_2(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_3(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_4(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_5(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_6(x)
		return x
        
	def _init_parameters(self):
		for m in list(self.linear):
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

class inv_dynamics_continuous(nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(inv_dynamics_continuous, self).__init__()
		self.linear_1 = nn.Linear(state_dim+state_dim, hidden_dim)
		self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_4 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_5 = nn.Linear(hidden_dim, hidden_dim)
		self.linear_6 = nn.Linear(hidden_dim, action_dim)
		self.linear = [self.linear_1, self.linear_2, self.linear_3, self.linear_4, self.linear_5, self.linear_6]
		self._init_parameters()

	def forward(self, s, a):
		x = torch.cat([s, a], dim=1)
		x = self.linear_1(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_2(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_3(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_4(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_5(x)
		x = F.leaky_relu(x, 0.001)
		x = self.linear_6(x)
		return x

	def _init_parameters(self):
		for m in list(self.linear):
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EnergyFunc(nn.Module):
    def __init__(self, state_dim=2, act_dim=2, h_dim=8, context_len=5):
        super(EnergyFunc, self).__init__()
        self.energyfunc = nn.Sequential(
            nn.Linear(context_len * state_dim + act_dim, h_dim),
            Swish(),
            nn.Linear(h_dim, h_dim),
            Swish(),
			nn.Linear(h_dim, h_dim),
            Swish(),
			nn.Linear(h_dim, h_dim),
            Swish(),
            nn.Linear(h_dim, 1)
        )
        self._init_parameters()
    
    def forward(self, states, actions):
        # x = torch.cat((states, actions), dim=-1).transpose(1,2).reshape(states.shape[0], -1)
        x = torch.cat((states.reshape(states.shape[0], -1), actions.reshape(actions.shape[0], -1)), dim=-1)
        return self.energyfunc(x)
    
    def _init_parameters(self):
        for m in list(self.energyfunc):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class EnergyFunc_cartpole(nn.Module):
    def __init__(self, state_dim=2, act_dim=2, h_dim=8, context_len=5):
        super(EnergyFunc_cartpole, self).__init__()
        self.energyfunc = nn.Sequential(
            nn.Linear(context_len * state_dim, context_len * h_dim),
            Swish(),
            nn.Linear(context_len * h_dim, context_len * h_dim),
            Swish(),
			nn.Linear(context_len * h_dim, context_len * h_dim),
            Swish(),
			nn.Linear(context_len * h_dim, context_len * h_dim),
            Swish(),
            nn.Linear(context_len * h_dim, 2)
        )
        self._init_parameters()
    
    def forward(self, states):
        # x = torch.cat((states, actions), dim=-1).transpose(1,2).reshape(states.shape[0], -1)
        # x = torch.cat((states.reshape(states.shape[0], -1), actions.reshape(actions.shape[0], -1)), dim=-1)
        return self.energyfunc(states.reshape(states.shape[0], -1))
    
    def _init_parameters(self):
        for m in list(self.energyfunc):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)