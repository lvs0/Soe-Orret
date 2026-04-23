"""
sampler/block_diffuser.py
Block-based diffusion sampler — 16 timesteps.
Applies iterative denoising in spatial blocks for memory efficiency.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class BlockDiffuser:
    """
    Iterative block diffusion with 16 timesteps.
    Processes image latent in spatial blocks to fit memory constraints.
    """

    def __init__(self, num_steps: int = 16, block_size: int = 32):
        self.num_steps = num_steps
        self.block_size = block_size
        self.timesteps = torch.linspace(1.0, 0.0, num_steps)

    def forward_blur(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Apply forward blur (Gaussian diffusion) at timestep t."""
        sigma = t * 0.5
        if sigma < 1e-3:
            return x
        kernel_size = int(6 * sigma + 1)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-coords**2 / (2 * sigma**2))
        g = g / g.sum()
        kernel_1d = g.view(1, 1, -1, 1)
        kernel_2d = kernel_1d * kernel_1d.transpose(2, 3)
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        x = F.conv2d(x, kernel_2d.to(x.device), padding=kernel_size // 2)
        return x.view_as(x)

    def denoise_step(self, x_noisy: torch.Tensor, model_output: torch.Tensor, t: float) -> torch.Tensor:
        """Single denoising step using model output."""
        alpha_t = 1.0 - t
        x_denoised = (x_noisy - (1 - alpha_t) * model_output) / alpha_t
        return x_denoised.clamp(-1.0, 1.0)

    def sample_blocks(
        self,
        latent_shape: Tuple[int, ...],
        score_fn,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Sample from noise using block-wise denoising."""
        B, C, H, W = latent_shape
        x = torch.randn(latent_shape, device=device)

        for i, t in enumerate(self.timesteps):
            # Process in blocks
            x_denoised = torch.zeros_like(x)
            count = torch.zeros_like(x)

            for by in range(0, H, self.block_size):
                for bx in range(0, W, self.block_size):
                    ey = min(by + self.block_size, H)
                    ex = min(bx + self.block_size, W)
                    block = x[:, :, by:ey, bx:ex]

                    # Simulate score model call
                    score = score_fn(block, t)
                    denoised = self.denoise_step(block, score, t.item())

                    x_denoised[:, :, by:ey, bx:ex] = denoised
                    count[:, :, by:ey, bx:ex] += 1.0

            x = x_denoised / count.clamp(min=1.0)

            if t.item() < 1e-3:
                break

        return x

    def __repr__(self) -> str:
        return f"BlockDiffuser(num_steps={self.num_steps}, block_size={self.block_size})"
