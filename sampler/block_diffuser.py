"""
Block Diffuser - Block-based diffusion sampler with 16 steps
Soe-Orret sampling module for structured generation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for block diffusion sampling"""
    num_steps: int = 16
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule: str = "linear"  # linear, cosine, quadratic
    block_size: int = 64
    num_blocks: int = 8


class BlockDiffuser:
    """
    Block-based diffusion sampler with configurable 16-step schedule.
    
    Implements DDPM-style diffusion with block-wise processing for
    efficient structured generation in the Soe-Orret system.
    """
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.betas = self._get_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Precompute diffusion constants
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate beta schedule for diffusion steps"""
        steps = self.config.num_steps
        if self.config.schedule == "linear":
            return torch.linspace(self.config.beta_start, self.config.beta_end, steps)
        elif self.config.schedule == "cosine":
            s = 0.008
            x = torch.linspace(0, steps, steps + 1)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif self.config.schedule == "quadratic":
            return torch.linspace(
                self.config.beta_start ** 0.5,
                self.config.beta_end ** 0.5,
                steps
            ) ** 2
        else:
            raise ValueError(f"Unknown schedule: {self.config.schedule}")
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process - add noise at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from noisy sample x_t"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance for reverse diffusion"""
        posterior_mean = (
            self._extract(self.betas, t, x_t.shape) * x_0 +
            self._extract(1.0 - self.alphas_cumprod_prev, t, x_t.shape) * x_t
        ) / self._extract(1.0 - self.alphas_cumprod, t, x_t.shape)
        
        posterior_variance = (
            self._extract(self.betas, t, x_t.shape) * 
            self._extract(1.0 - self.alphas_cumprod_prev, t, x_t.shape) / 
            self._extract(1.0 - self.alphas_cumprod, t, x_t.shape)
        )
        
        return posterior_mean, posterior_variance
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Extract values at timesteps and reshape"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)
    
    def p_sample(self, model: Callable, x_t: torch.Tensor, t: torch.Tensor, 
                 condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single reverse diffusion step (denoising)"""
        # Predict noise
        with torch.no_grad():
            noise_pred = model(x_t, t, condition)
        
        # Predict x_0
        x_0_pred = self.predict_start_from_noise(x_t, t, noise_pred)
        
        # Compute posterior
        model_mean, model_variance = self.q_posterior_mean_variance(x_0_pred, x_t, t)
        
        # Sample
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(model_variance) * noise
    
    def sample_blocks(self, model: Callable, shape: Tuple[int, ...], 
                      condition: Optional[torch.Tensor] = None,
                      device: str = "cpu") -> torch.Tensor:
        """
        Generate samples using block-wise diffusion (16 steps)
        
        Args:
            model: Noise prediction model
            shape: Output shape (batch, channels, height, width)
            condition: Optional conditioning tensor
            device: Device to run on
            
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        block_size = self.config.block_size
        num_blocks = self.config.num_blocks
        
        # Initialize from noise
        x = torch.randn(shape, device=device)
        
        # Process in blocks
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, shape[-1])
            
            # Reverse diffusion for this block
            for step in reversed(range(self.config.num_steps)):
                t = torch.full((batch_size,), step, device=device, dtype=torch.long)
                
                # Extract block
                if len(shape) == 4:
                    x_block = x[:, :, :, block_start:block_end]
                else:
                    x_block = x[:, block_start:block_end]
                
                # Denoise step
                x_block = self.p_sample(model, x_block, t, condition)
                
                # Place back
                if len(shape) == 4:
                    x[:, :, :, block_start:block_end] = x_block
                else:
                    x[:, block_start:block_end] = x_block
        
        return x
    
    def sample(self, model: Callable, shape: Tuple[int, ...],
               condition: Optional[torch.Tensor] = None,
               device: str = "cpu") -> torch.Tensor:
        """
        Standard 16-step diffusion sampling without blocks
        
        Args:
            model: Noise prediction model
            shape: Output shape
            condition: Optional conditioning tensor
            device: Device to run on
            
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        for step in reversed(range(self.config.num_steps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition)
        
        return x
    
    def training_loss(self, model: Callable, x_0: torch.Tensor, 
                      condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute training loss for a batch"""
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Random timesteps
        t = torch.randint(0, self.config.num_steps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = model(x_t, t, condition)
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        return loss


class SimpleNoiseModel(nn.Module):
    """Simple U-Net style noise prediction model for testing"""
    
    def __init__(self, channels: int = 3, time_emb_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128 + time_emb_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / 16.0)
        t_emb = t_emb.view(t_emb.shape[0], t_emb.shape[1], 1, 1)
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
        
        # Encode
        h = self.encoder(x)
        
        # Concatenate time embedding
        h = torch.cat([h, t_emb], dim=1)
        
        # Decode
        return self.decoder(h)


if __name__ == "__main__":
    # Test the diffuser
    config = DiffusionConfig(num_steps=16, block_size=32, num_blocks=4)
    diffuser = BlockDiffuser(config)
    model = SimpleNoiseModel()
    
    # Test sampling
    samples = diffuser.sample(model, (4, 3, 64, 64))
    print(f"Generated samples shape: {samples.shape}")
    
    # Test block sampling
    block_samples = diffuser.sample_blocks(model, (2, 3, 64, 64))
    print(f"Block samples shape: {block_samples.shape}")
