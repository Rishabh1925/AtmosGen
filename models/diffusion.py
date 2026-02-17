import torch
import torch.nn as nn
import numpy as np

class Diffusion:

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):

        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        noise = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def sample_timestamps(self, n):
        return torch.randint(low=0, high=self.timesteps, size=(n,))