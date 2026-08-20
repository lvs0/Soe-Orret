"""
Block Diffuser - Block-based diffusion sampler with 16 steps
"""

import numpy as np
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion sampler."""
    num_steps: int = 16
    beta_start: float = 0.0001
    beta_end: float = 0.02
    block_size: int = 4
    noise_scale: float = 1.0


class BlockDiffuser:
    """
    Block-based diffusion sampler implementing a DDPM-like process
    with configurable steps and block-wise processing.
    """
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)
    
    def _compute_betas(self) -> np.ndarray:
        """Compute linear beta schedule."""
        return np.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_steps
        )
    
    def add_noise(self, x: np.ndarray, t: int, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Add noise to input at timestep t.
        
        Args:
            x: Clean input array
            t: Timestep index
            noise: Optional pre-generated noise
            
        Returns:
            Noised version of x
        """
        if noise is None:
            noise = np.random.randn(*x.shape) * self.config.noise_scale
        
        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]
        
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
    
    def denoise_step(self, x_t: np.ndarray, predicted_noise: np.ndarray, t: int) -> np.ndarray:
        """
        Single denoising step.
        
        Args:
            x_t: Current noised state
            predicted_noise: Noise predicted by model
            t: Current timestep
            
        Returns:
            Slightly denoised state
        """
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]
        
        # Compute the predicted x_0
        sqrt_alpha_bar = np.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
        
        # Compute posterior mean
        alpha_bar_prev = self.alpha_bars[t - 1] if t > 0 else 1.0
        coef1 = np.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
        coef2 = np.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        
        mean = coef1 * x_0_pred + coef2 * x_t
        
        # Add noise if not final step
        if t > 0:
            noise = np.random.randn(*x_t.shape) * np.sqrt(beta)
            return mean + noise
        
        return mean
    
    def sample_blocks(self, shape: Tuple[int, ...], model: Callable[[np.ndarray, int], np.ndarray]) -> np.ndarray:
        """
        Generate samples using block-based processing.
        
        Args:
            shape: Output shape
            model: Noise prediction model (takes x and t, returns predicted noise)
            
        Returns:
            Generated sample
        """
        # Start from pure noise
        x = np.random.randn(*shape) * self.config.noise_scale
        
        # Iteratively denoise
        for t in reversed(range(self.config.num_steps)):
            predicted_noise = model(x, t)
            x = self.denoise_step(x, predicted_noise, t)
        
        return x
    
    def process_in_blocks(self, data: np.ndarray, block_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Process data in blocks for memory efficiency.
        
        Args:
            data: Input data array
            block_fn: Function to apply to each block
            
        Returns:
            Processed data
        """
        block_size = self.config.block_size
        shape = data.shape
        
        # Calculate number of blocks per dimension
        num_blocks = [max(1, s // block_size) for s in shape]
        
        result = np.zeros_like(data)
        
        # Process each block
        for idx in np.ndindex(*num_blocks):
            # Calculate slice indices
            slices = []
            for i, (dim_idx, dim_size) in enumerate(zip(idx, shape)):
                start = dim_idx * block_size
                end = min(start + block_size, dim_size)
                slices.append(slice(start, end))
            
            # Extract, process, and store block
            block = data[tuple(slices)]
            processed_block = block_fn(block)
            result[tuple(slices)] = processed_block
        
        return result
    
    def forward_process(self, x_0: np.ndarray) -> List[np.ndarray]:
        """
        Run forward diffusion process, returning all intermediate states.
        
        Args:
            x_0: Clean input
            
        Returns:
            List of states from x_0 to x_T
        """
        states = [x_0]
        x = x_0
        
        for t in range(self.config.num_steps):
            x = self.add_noise(x, t)
            states.append(x.copy())
        
        return states


def create_simple_model(shape: Tuple[int, ...]) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Create a simple noise prediction model for testing.
    
    Args:
        shape: Expected input shape
        
    Returns:
        Model function
    """
    def model(x: np.ndarray, t: int) -> np.ndarray:
        # Simple linear model: predict scaled noise
        scale = 1.0 - (t / 16.0)  # Decreasing scale over time
        return np.random.randn(*x.shape) * 0.1 * scale
    
    return model


if __name__ == "__main__":
    # Example usage
    config = DiffusionConfig(num_steps=16, block_size=4)
    diffuser = BlockDiffuser(config)
    
    # Test forward process
    x_0 = np.random.randn(8, 8)
    states = diffuser.forward_process(x_0)
    print(f"Forward process: {len(states)} states")
    print(f"Initial shape: {states[0].shape}, Final shape: {states[-1].shape}")
    
    # Test sampling
    model = create_simple_model((8, 8))
    sample = diffuser.sample_blocks((8, 8), model)
    print(f"Generated sample shape: {sample.shape}")
    
    # Test block processing
    data = np.random.randn(16, 16)
    result = diffuser.process_in_blocks(data, lambda b: b * 0.5)
    print(f"Block processed shape: {result.shape}")
