import torch

class Diffusion:
    def __init__(self, timesteps=500):
        self.timesteps = timesteps
        
        # Create noise schedule
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def sample_timesteps(self, batch_size):
        """Sample random timesteps for training"""
        return torch.randint(0, self.timesteps, (batch_size,))
    
    def add_noise(self, x, t):
        """Add noise to images at timestep t"""
        noise = torch.randn_like(x)
        
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]
        
        noisy_x = torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * noise
        
        return noisy_x, noise