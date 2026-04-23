"""Block diffusion sampler with DDPM-style 16-step schedule."""

import torch
import numpy as np
from typing import Literal
from dataclasses import dataclass


@dataclass
class SamplerConfig:
    num_steps: int = 16
    beta_schedule: Literal["linear", "cosine", "quadratic"] = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BlockDiffuser:
    """Block-based diffusion sampler for DDPM-style generative modeling.

    Implements forward diffusion (corrupting data) and reverse diffusion
    (denoising) with configurable beta schedules and block-wise processing.
    """

    def __init__(self, config: SamplerConfig | None = None):
        self.config = config or SamplerConfig()
        self.device = self.config.device
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def _get_betas(self) -> torch.Tensor:
        schedule = self.config.beta_schedule
        T = self.config.num_steps
        betas = torch.zeros(T)
        t = torch.arange(T) / (T - 1)

        if schedule == "linear":
            betas = self.config.beta_start + t * (self.config.beta_end - self.config.beta_start)
        elif schedule == "cosine":
            betas = self.config.beta_start + 0.5 * (self.config.beta_end - self.config.beta_start) * (1 - torch.cos(np.pi * t))
        elif schedule == "quadratic":
            betas = self.config.beta_start + t.pow(2) * (self.config.beta_end - self.config.beta_start)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        return betas.to(self.device)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to data at timestep t.

        Args:
            x0: Original data [B, C, H, W]
            t: Timestep indices [B,]

        Returns:
            (noisy_x, noise) at timestep t
        """
        batch_size = x0.shape[0]
        alpha_bar_t = self.alpha_bar[t].reshape(batch_size, 1, 1, 1)
        noise = torch.randn_like(x0)
        noisy_x = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise

    def reverse_step(self, x_t: torch.Tensor, t: int, block_idx: int | None = None) -> torch.Tensor:
        """Single reverse diffusion step.

        Args:
            x_t: Noisy data at step t
            t: Current timestep (0-indexed, t=0 is clean)
            block_idx: Optional block index for block-wise processing
        """
        T = self.config.num_steps
        effective_t = T - 1 - t

        alpha_t = self.alphas[effective_t]
        alpha_bar_t = self.alpha_bar[effective_t]
        beta_t = self.betas[effective_t]

        beta_t = beta_t.reshape(1, 1, 1, 1)
        alpha_t = alpha_t.reshape(1, 1, 1, 1)
        alpha_bar_t = alpha_bar_t.reshape(1, 1, 1, 1)

        predicted_noise = x_t * 0.1
        model_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t + 1e-8)

        mean = (x_t - beta_t * model_pred) / torch.sqrt(alpha_t + 1e-8)
        variance = beta_t

        if t > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance + 1e-8) * noise
        return mean

    def sample(self, shape: tuple, init_x: torch.Tensor | None = None) -> torch.Tensor:
        """Generate samples via reverse diffusion.

        Args:
            shape: Sample shape [B, C, H, W]
            init_x: Optional starting noise (uses random if None)

        Returns:
            Generated samples
        """
        B, C, H, W = shape
        x_t = init_x if init_x is not None else torch.randn(shape, device=self.device)

        for step in range(self.config.num_steps):
            x_t = self.reverse_step(x_t, step)

        return torch.clamp(x_t, -1.0, 1.0)

    def block_sample(self, shape: tuple, block_size: int = 4) -> torch.Tensor:
        """Generate samples with block-wise processing for efficiency.

        Args:
            shape: Sample shape [B, C, H, W]
            block_size: Size of processing blocks

        Returns:
            Generated samples with block-wise refinement
        """
        B, C, H, W = shape
        x_t = torch.randn(shape, device=self.device)

        for step in range(self.config.num_steps):
            block_count = 0
            for i in range(0, H, block_size):
                for j in range(0, W, block_size):
                    x_block = x_t[:, :, i:i+block_size, j:j+block_size]
                    x_block = self.reverse_step(x_block, step, block_idx=block_count)
                    x_t[:, :, i:i+block_size, j:j+block_size] = x_block
                    block_count += 1

        return torch.clamp(x_t, -1.0, 1.0)
