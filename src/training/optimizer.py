"""
Memory-efficient optimizers for large model training.
"""
import torch
from typing import Dict, Any, List, Optional
import logging

try:
    from bitsandbytes.optim import AdamW8bit
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes not available, falling back to standard optimizers")

logger = logging.getLogger(__name__)

def create_optimizer(model, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer with memory-efficient options.
    
    Args:
        model: The model to optimize
        config: Training configuration dictionary
        
    Returns:
        Configured optimizer
    """
    # Separate parameters by type for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Parameters that should not have weight decay
        if any(nd in name for nd in ["bias", "LayerNorm", "layernorm", "layer_norm", "ln", "bn"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": config.get('weight_decay', 0.01),
            "lr": config.get('learning_rate', 1e-4)
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": config.get('learning_rate', 1e-4)
        },
    ]
    
    # Log parameter counts
    decay_count = sum(p.numel() for p in decay_params)
    no_decay_count = sum(p.numel() for p in no_decay_params)
    logger.info(f"Optimizer: {decay_count/1e6:.2f}M params with decay, {no_decay_count/1e6:.2f}M without decay")
    
    # Choose optimizer based on configuration
    optimizer_type = config.get('optimizer_type', 'adamw')
    use_8bit_adam = config.get('use_8bit_adam', False) and BITSANDBYTES_AVAILABLE
    
    if use_8bit_adam:
        logger.info("Using 8-bit AdamW optimizer for memory efficiency")
        optimizer = AdamW8bit(
            optimizer_grouped_parameters,
            lr=config.get('learning_rate', 1e-4),
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
            eps=config.get('adam_epsilon', 1e-8),
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.get('learning_rate', 1e-4),
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
            eps=config.get('adam_epsilon', 1e-8),
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=config.get('learning_rate', 1e-4),
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
            eps=config.get('adam_epsilon', 1e-8),
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=config.get('learning_rate', 1e-4),
            momentum=config.get('momentum', 0.9),
            nesterov=config.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer.__class__.__name__} optimizer with lr={config.get('learning_rate', 1e-4)}")
    
    return optimizer

def create_scheduler(optimizer, config: Dict[str, Any], num_training_steps: int):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        config: Training configuration
        num_training_steps: Total number of training steps
        
    Returns:
        Configured scheduler
    """
    scheduler_type = config.get('scheduler_type', 'cosine_with_warmup')
    warmup_steps = config.get('warmup_steps', 1000)
    
    if scheduler_type == 'cosine_with_warmup':
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'linear_with_warmup':
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'constant_with_warmup':
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_type == 'polynomial':
        from transformers import get_polynomial_decay_schedule_with_warmup
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            power=config.get('polynomial_power', 1.0)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 10000),
            gamma=config.get('gamma', 0.5)
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.get('min_lr', 1e-6)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler.__class__.__name__} scheduler with {warmup_steps} warmup steps")
    
    return scheduler

class GradientClippingMixin:
    """Mixin for gradient clipping utilities."""
    
    @staticmethod
    def clip_grad_norm(model, max_norm: float, norm_type: float = 2.0) -> float:
        """
        Clip gradient norm.
        
        Args:
            model: Model to clip gradients for
            max_norm: Maximum gradient norm
            norm_type: Type of norm to compute
            
        Returns:
            Total norm of the parameters (before clipping)
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        device = parameters[0].grad.device
        
        if norm_type == float('inf'):
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                norm_type
            )
        
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef.to(p.grad.device))
        
        return total_norm.item()

class OptimizerWrapper:
    """
    Wrapper for optimizer with additional utilities.
    """
    
    def __init__(self, optimizer, config: Dict[str, Any]):
        self.optimizer = optimizer
        self.config = config
        self.gradient_clipper = GradientClippingMixin()
        
        # State tracking
        self.step_count = 0
        self.total_norm_history = []
        
    def step(self, model=None):
        """Step the optimizer with optional gradient clipping."""
        # Gradient clipping
        max_grad_norm = self.config.get('max_grad_norm', 0)
        if max_grad_norm > 0 and model is not None:
            total_norm = self.gradient_clipper.clip_grad_norm(model, max_grad_norm)
            self.total_norm_history.append(total_norm)
            
            # Keep only recent history
            if len(self.total_norm_history) > 1000:
                self.total_norm_history = self.total_norm_history[-1000:]
        
        # Optimizer step
        self.optimizer.step()
        self.step_count += 1
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Get optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.total_norm_history:
            return {}
        
        import numpy as np
        norms = np.array(self.total_norm_history)
        
        return {
            'gradient_norm_mean': float(np.mean(norms)),
            'gradient_norm_std': float(np.std(norms)),
            'gradient_norm_max': float(np.max(norms)),
            'gradient_norm_min': float(np.min(norms)),
            'gradient_norm_recent': float(norms[-1]) if len(norms) > 0 else 0.0
        }