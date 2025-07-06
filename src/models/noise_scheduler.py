"""
Noise scheduling for diffusion models.
"""
import torch
import numpy as np
from typing import Optional, Union, List, Tuple
import math

class MaskedDiffusionScheduler:
    """
    Implements noise scheduling for masked diffusion language models.
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        mask_token_id: int = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        prediction_type: str = "epsilon"
    ):
        """
        Initialize the noise scheduler.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            schedule_type: Type of noise schedule ("cosine", "linear", "scaled_linear")
            mask_token_id: Token ID used for masking
            beta_start: Start value for beta schedule
            beta_end: End value for beta schedule
            clip_sample: Whether to clip samples to [-1, 1]
            prediction_type: Type of prediction ("epsilon", "sample", "v_prediction")
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.mask_token_id = mask_token_id
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        
        # Create noise schedule
        if schedule_type == "cosine":
            self.betas = self._cosine_schedule()
        elif schedule_type == "linear":
            self.betas = self._linear_schedule()
        elif schedule_type == "scaled_linear":
            self.betas = self._scaled_linear_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For numerical stability
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Log calculation clipped because the posterior variance is 0 at the beginning
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # Set timesteps
        self.timesteps = torch.arange(0, num_timesteps).flip(dims=[0])
        
    def _cosine_schedule(self) -> torch.Tensor:
        """Cosine noise schedule."""
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
        
        betas = []
        for i in range(self.num_timesteps):
            t1 = i / self.num_timesteps
            t2 = (i + 1) / self.num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float32)
    
    def _linear_schedule(self) -> torch.Tensor:
        """Linear noise schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32)
    
    def _scaled_linear_schedule(self) -> torch.Tensor:
        """Scaled linear noise schedule."""
        return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_timesteps, dtype=torch.float32) ** 2
    
    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Scale the input sample for the model.
        
        Args:
            sample: Input sample
            timestep: Current timestep
            
        Returns:
            Scaled sample
        """
        return sample
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        timesteps: torch.Tensor,
        noise_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to input sequences by masking tokens.
        
        Args:
            original_samples: Original token sequences of shape (batch_size, seq_len)
            timesteps: Timesteps for each sample of shape (batch_size,)
            noise_ratio: Optional fixed noise ratio instead of schedule-based
            
        Returns:
            Tuple of (noisy_samples, mask) where mask indicates which tokens were masked
        """
        if self.mask_token_id is None:
            raise ValueError("mask_token_id must be set to use add_noise")
            
        batch_size, seq_len = original_samples.shape
        device = original_samples.device
        
        # Get noise rates for each timestep
        if noise_ratio is not None:
            # Use fixed noise ratio
            noise_rates = torch.full((batch_size,), noise_ratio, device=device)
        else:
            # Use schedule-based noise rates
            noise_rates = 1.0 - self.alphas_cumprod[timesteps].to(device)
        
        # Create random mask based on noise rates
        rand = torch.rand(batch_size, seq_len, device=device)
        mask = rand < noise_rates.unsqueeze(1)
        
        # Apply mask - replace masked tokens with mask_token_id
        noisy_samples = original_samples.clone()
        noisy_samples[mask] = self.mask_token_id
        
        return noisy_samples, mask
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get velocity for v-parameterization.
        
        Args:
            sample: Original sample
            noise: Noise tensor
            timesteps: Timesteps
            
        Returns:
            Velocity tensor
        """
        alphas_cumprod = self.alphas_cumprod[timesteps]
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()
        
        velocity = sqrt_alphas_cumprod * noise - sqrt_one_minus_alphas_cumprod * sample
        return velocity
    
    def get_loss_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get loss weights for different timesteps.
        
        Args:
            timesteps: Timesteps tensor
            
        Returns:
            Loss weights tensor
        """
        # SNR weighting
        snr = self.alphas_cumprod[timesteps] / (1.0 - self.alphas_cumprod[timesteps])
        
        # Different weighting strategies
        if self.prediction_type == "epsilon":
            # Standard epsilon weighting
            weight = 1.0 / (1.0 + snr)
        elif self.prediction_type == "v_prediction":
            # v-prediction weighting
            weight = 1.0 / (snr + 1)
        else:
            # Uniform weighting
            weight = torch.ones_like(snr)
        
        return weight
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, dict]:
        """
        Predict the sample at the previous timestep.
        
        Args:
            model_output: Direct output from learned diffusion model
            timestep: Current discrete timestep in diffusion chain
            sample: Current instance of sample being created by diffusion process
            generator: Random number generator
            return_dict: Whether to return dict or tensor
            
        Returns:
            Previous sample or dict with 'prev_sample'
        """
        t = timestep
        
        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 2. Compute predicted original sample from predicted noise
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"prediction_type {self.prediction_type} not supported")
        
        # 3. Clip or clamp predicted original sample
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 4. Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (
            alpha_prod_t_prev.sqrt() * self.betas[t] / beta_prod_t
        )
        current_sample_coeff = self.alphas[t].sqrt() * beta_prod_t_prev / beta_prod_t
        
        # 5. Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
            variance = self.posterior_variance[t].sqrt() * variance_noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        if not return_dict:
            return pred_prev_sample
        
        return {"prev_sample": pred_prev_sample, "pred_original_sample": pred_original_sample}
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Set the discrete timesteps used for inference.
        
        Args:
            num_inference_steps: Number of diffusion steps used when generating samples
            device: Device to put the timesteps on
        """
        if num_inference_steps > self.num_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) must be <= num_timesteps ({self.num_timesteps})"
            )
        
        self.num_inference_steps = num_inference_steps
        
        # Create timesteps
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        
        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    def previous_timestep(self, timestep: int) -> int:
        """Get the previous timestep."""
        prev_t = timestep - self.num_timesteps // self.num_inference_steps
        return prev_t
    
    def __len__(self) -> int:
        """Return the number of timesteps."""
        return self.num_timesteps