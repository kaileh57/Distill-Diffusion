"""
Hardware optimization utilities.
"""
import torch
import psutil
import subprocess
import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    gpu_info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        memory_allocated = torch.cuda.memory_allocated(i) / 1e9
        memory_reserved = torch.cuda.memory_reserved(i) / 1e9
        
        gpu_info["devices"].append({
            "id": i,
            "name": props.name,
            "total_memory_gb": memory_total,
            "allocated_memory_gb": memory_allocated,
            "reserved_memory_gb": memory_reserved,
            "free_memory_gb": memory_total - memory_reserved,
            "compute_capability": f"{props.major}.{props.minor}"
        })
    
    return gpu_info

def setup_hardware(config: Dict[str, Any]) -> str:
    """
    Setup hardware configuration.
    
    Args:
        config: Hardware configuration dictionary
        
    Returns:
        Device string to use
    """
    # Check GPU availability
    gpu_info = get_gpu_info()
    
    if not gpu_info["available"]:
        logger.warning("No GPU available, using CPU")
        return "cpu"
    
    # Log GPU info
    logger.info(f"Found {gpu_info['count']} GPU(s)")
    for device in gpu_info["devices"]:
        logger.info(f"GPU {device['id']}: {device['name']} ({device['total_memory_gb']:.1f}GB)")
    
    # Set device
    device = config.get('device', 'cuda:0')
    if device.startswith('cuda'):
        device_id = int(device.split(':')[1]) if ':' in device else 0
        if device_id >= gpu_info["count"]:
            logger.warning(f"Requested GPU {device_id} not available, using GPU 0")
            device = "cuda:0"
    
    # Set memory fraction if specified
    max_memory_gb = config.get('max_memory_gb', None)
    if max_memory_gb and device.startswith('cuda'):
        device_id = int(device.split(':')[1]) if ':' in device else 0
        total_memory = gpu_info["devices"][device_id]["total_memory_gb"]
        fraction = max_memory_gb / total_memory
        torch.cuda.set_per_process_memory_fraction(fraction, device_id)
        logger.info(f"Set memory fraction to {fraction:.2f} ({max_memory_gb}GB)")
    
    # Configure torch settings
    if config.get('compile_model', False):
        try:
            # Enable torch.compile for PyTorch 2.0+
            torch.set_float32_matmul_precision('high')
            logger.info("Enabled torch.compile optimizations")
        except Exception as e:
            logger.warning(f"Could not enable torch.compile: {e}")
    
    # Set deterministic operations if requested
    if config.get('deterministic', False):
        torch.use_deterministic_algorithms(True)
        logger.info("Enabled deterministic algorithms")
    
    return device

def monitor_memory_usage(device: str = "cuda:0") -> Dict[str, float]:
    """Monitor current memory usage."""
    if not torch.cuda.is_available() or device == "cpu":
        return {"available": False}
    
    device_id = int(device.split(':')[1]) if ':' in device else 0
    
    memory_allocated = torch.cuda.memory_allocated(device_id) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device_id) / 1e9
    memory_total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
    
    return {
        "available": True,
        "allocated_gb": memory_allocated,
        "reserved_gb": memory_reserved,
        "total_gb": memory_total,
        "free_gb": memory_total - memory_reserved,
        "utilization": memory_reserved / memory_total
    }

def clear_gpu_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache")

def estimate_model_memory(num_parameters: int, dtype: str = "float16") -> Dict[str, float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        num_parameters: Number of model parameters
        dtype: Data type (float16, float32, etc.)
        
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = {
        "float16": 2,
        "float32": 4,
        "int8": 1
    }
    
    param_bytes = bytes_per_param.get(dtype, 4)
    
    # Basic estimates
    weights_gb = num_parameters * param_bytes / 1e9
    
    # Training memory estimates (rule of thumb)
    gradients_gb = weights_gb  # Same as weights
    optimizer_states_gb = weights_gb * 2  # Adam has 2 momentum terms
    activations_gb = weights_gb * 2  # Rough estimate
    
    training_total = weights_gb + gradients_gb + optimizer_states_gb + activations_gb
    
    return {
        "weights_gb": weights_gb,
        "gradients_gb": gradients_gb,
        "optimizer_states_gb": optimizer_states_gb,
        "activations_gb": activations_gb,
        "training_total_gb": training_total,
        "inference_gb": weights_gb + activations_gb * 0.5
    }

def check_memory_requirements(model, config: Dict[str, Any]) -> bool:
    """
    Check if model fits in available memory.
    
    Args:
        model: PyTorch model
        config: Hardware configuration
        
    Returns:
        True if model should fit, False otherwise
    """
    num_params = sum(p.numel() for p in model.parameters())
    dtype = config.get('dtype', 'float16')
    
    memory_est = estimate_model_memory(num_params, dtype)
    gpu_info = get_gpu_info()
    
    if not gpu_info["available"]:
        logger.warning("No GPU available for memory check")
        return False
    
    device_id = 0  # Assume using first GPU
    available_memory = gpu_info["devices"][device_id]["total_memory_gb"]
    required_memory = memory_est["training_total_gb"]
    
    logger.info(f"Estimated memory requirement: {required_memory:.2f}GB")
    logger.info(f"Available GPU memory: {available_memory:.2f}GB")
    
    if required_memory > available_memory * 0.9:  # 90% threshold
        logger.warning("Model may not fit in GPU memory!")
        logger.warning("Consider:")
        logger.warning("- Using gradient checkpointing")
        logger.warning("- Reducing batch size")
        logger.warning("- Using 8-bit optimizers")
        logger.warning("- Using LoRA/QLoRA")
        return False
    
    return True

def optimize_for_hardware(model, config: Dict[str, Any]):
    """
    Apply hardware-specific optimizations.
    
    Args:
        model: PyTorch model
        config: Hardware configuration
    """
    # Enable gradient checkpointing if requested
    if config.get('gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
    
    # Compile model if requested and supported
    if config.get('compile_model', False):
        try:
            model = torch.compile(model)
            logger.info("Compiled model with torch.compile")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    # Set memory efficient attention if available
    if config.get('use_flash_attention', False):
        try:
            # This is a placeholder - actual implementation depends on model type
            logger.info("Flash attention requested but not implemented")
        except Exception as e:
            logger.warning(f"Could not enable flash attention: {e}")
    
    return model

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        "memory_total_gb": psutil.virtual_memory().total / 1e9,
        "memory_available_gb": psutil.virtual_memory().available / 1e9,
        "gpu_info": get_gpu_info(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }