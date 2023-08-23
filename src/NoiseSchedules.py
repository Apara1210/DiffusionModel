import torch

def beta_linear_schedule(timesteps, start, end):
    return torch.linspace(start, end, timesteps)

def get_noise_schedule(timesteps, start= 0.0001, end= 0.02, schedule= 'Linear'):
    if schedule == 'Linear':
        return beta_linear_schedule(timesteps, start, end)