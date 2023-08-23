import torch
import sys
sys.path.append('..')

import utils
from config import config

class ForwardDiffusion():
    # Forward diffusion process
    def __init__(self, betas, device= config['device']):
        ## with the noise schedule beta, the init calculates all alphas
        self.betas  = betas

        self.alphas             = 1 - betas
        self.alphas_bar         = torch.cumprod(self.alphas, axis= 0)
        self.device             = device

        self.alphas_bar_sqrt    = self.alphas_bar**0.5
        self.alphas_bar_sqrt_1  = (1 - self.alphas_bar)**0.5

    def add_noise(self, x_0, t):
        """
        x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * eps (Reparameterization trick)
        Takes input the sample x_0 and timestep t, returns x_t and associated noise
        """
        x_0         = x_0.to(self.device)
        noise       = torch.randn_like(x_0).to(self.device)

        alphas_1_t  = utils.get_index_from_list(self.alphas_bar_sqrt, t, x_0.shape).to(self.device)
        alphas_2_t  = utils.get_index_from_list(self.alphas_bar_sqrt_1, t, x_0.shape).to(self.device)

        x_t         = alphas_1_t * x_0 + alphas_2_t * noise

        return x_t, noise