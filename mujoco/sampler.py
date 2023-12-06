import torch
import torch.nn.functional as F

from math import sqrt

action_delta = torch.ones(1, device='cuda') * 1.


def prior_sample_new_exmps(energy_model,
                           sample_size, 
                           states, 
                           steps, 
                           act_dim,
                           stepsize_init,
                           stepsize_end,
                           init_act=None, 
                           add_noise=False, 
                           noise_scale=0.5,
                           optimal=False):
    is_training = energy_model.training
    energy_model.eval()
    for p in energy_model.parameters():
        p.requires_grad = False
    had_gradients_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)

    schedule = PolynomialSchedule(stepsize_init, stepsize_end, 2.0, steps)

    if init_act is None:
        inp_actions = torch.rand((sample_size, 1, act_dim), device=states.device) * 2 * action_delta - action_delta
    else:
        inp_actions = init_act.detach()
    inp_actions.requires_grad = True

    noise_0 = torch.randn(inp_actions.shape, device=inp_actions.device)

    energy_min = torch.ones(sample_size, device=inp_actions.device) * torch.inf
    action_min = torch.zeros_like(inp_actions)

    step_size = stepsize_init
    for step in range(1, steps+1):

        delta_y = 0.
        if add_noise:
            noise_0.normal_(0, noise_scale)
            # delta_y += noise_0.data
            inp_actions.data.add_(noise_0.data)
            inp_actions.data.clamp_(min=-action_delta, max=action_delta)
        
        energy = -energy_model(states, inp_actions).squeeze(-1)
        
        action_min[energy<energy_min]=inp_actions[energy<energy_min]
        energy_min[energy<energy_min]=energy[energy<energy_min]
        
        energy.sum().backward()
        inp_actions.grad.data.clamp_(-0.3, 0.3)
        delta_y -= inp_actions.grad.data * 0.5
        delta_y *= step_size
        # delta_y.data.clamp_(-0.5, 0.5)

        inp_actions.data.add_(delta_y)
        # inp_actions.data.add_(-step_size * inp_actions.grad.data)
        inp_actions.grad.detach_()
        inp_actions.grad.zero_()
        inp_actions.data.clamp_(min=-action_delta, max=action_delta)

        step_size = schedule.get_rate(step + 1)

    for p in energy_model.parameters():
        p.requires_grad = True
    energy_model.train(is_training)
    torch.set_grad_enabled(had_gradients_enabled)
    
    if optimal:
        return action_min.detach()
    else:
        return inp_actions.detach()



class PolynomialSchedule:
    def __init__(self, init, final, power=2.0, num_steps=100):
        self._init = init
        self._final = final
        self._power = power
        self._num_steps = num_steps

    def get_rate(self, index):
        return ((self._init - self._final) * ((1 - (float(index) / float(self._num_steps - 1))) ** (self._power))) + self._final


def get_probabilities(energy_model,
                      batch_size,
                      num_action_samples,
                      states,
                      actions,
                      temperature=1.0):
    net_logits = energy_model(states, actions).reshape(batch_size, num_action_samples)
    probs = F.softmax(net_logits / temperature, dim=1)
    return probs 