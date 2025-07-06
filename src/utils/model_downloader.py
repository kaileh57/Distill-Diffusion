"""
Download and analyze pre-trained models to determine conversion feasibility.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, cache_dir="./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def download_model(self, model_name, model_configs=None):
        """
        Download model and analyze its architecture for conversion.
        
        Args:
            model_name: HuggingFace model identifier
            model_configs: Dict with target configs for conversion
        """
        print(f"Downloading {model_name}...")
        
        # Download model configuration first
        config = AutoConfig.from_pretrained(model_name)
        
        # Download model and tokenizer
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        except Exception as e:
            logger.warning(f"Failed to download with device_map=auto: {e}")
            # Fallback to CPU loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            tokenizer = None
        
        # Analyze model architecture
        analysis = self.analyze_model(model, config)
        
        # Save analysis
        analysis_file = self.cache_dir / f"{model_name.replace('/', '_')}_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return model, tokenizer, analysis
    
    def analyze_model(self, model, config):
        """Analyze model architecture for conversion requirements."""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Try to get layer count from different model architectures
        num_layers = 0
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            num_layers = len(model.transformer.h)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        elif hasattr(config, 'num_hidden_layers'):
            num_layers = config.num_hidden_layers
        elif hasattr(config, 'n_layer'):
            num_layers = config.n_layer
        
        # Get architecture details
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', 0))
        num_attention_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 0))
        vocab_size = getattr(config, 'vocab_size', 0)
        
        analysis = {
            "model_name": config.name_or_path if hasattr(config, 'name_or_path') else "unknown",
            "model_type": getattr(config, 'model_type', 'unknown'),
            "total_parameters": total_params,
            "total_parameters_billions": total_params / 1e9,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "vocab_size": vocab_size,
            "max_position_embeddings": getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 0)),
            "memory_requirements": {
                "fp16_weights_gb": total_params * 2 / 1e9,
                "fp32_weights_gb": total_params * 4 / 1e9,
                "training_estimate_gb": total_params * 8 / 1e9,  # Conservative estimate
                "inference_gb": total_params * 2.5 / 1e9
            },
            "conversion_feasibility": self._assess_conversion_feasibility(total_params, config)
        }
        
        return analysis
    
    def _assess_conversion_feasibility(self, total_params, config):
        """Assess if model can be converted given hardware constraints."""
        param_billions = total_params / 1e9
        
        # Hardware assumptions: RTX A6000 (48GB) + RTX A4500 (20GB) = 68GB total
        available_memory = 68  # GB
        
        # Estimate memory requirements
        training_memory = total_params * 8 / 1e9  # 8 bytes per param for full training
        
        feasibility = {
            "can_fit_single_gpu": training_memory < 45,  # Leave headroom on A6000
            "requires_multi_gpu": training_memory > 45,
            "memory_optimizations_needed": training_memory > 30,
            "recommended_optimizations": []
        }
        
        if training_memory > 45:
            feasibility["recommended_optimizations"].append("gradient_checkpointing")
        if training_memory > 30:
            feasibility["recommended_optimizations"].append("8bit_adam")
        if param_billions > 3:
            feasibility["recommended_optimizations"].append("lora_or_qlora")
        if training_memory > 60:
            feasibility["recommended_optimizations"].append("deepspeed_zero")
        
        return feasibility
    
    def get_model_info(self, model_name):
        """Get basic model information without downloading."""
        try:
            config = AutoConfig.from_pretrained(model_name)
            return {
                "model_type": getattr(config, 'model_type', 'unknown'),
                "hidden_size": getattr(config, 'hidden_size', getattr(config, 'd_model', 0)),
                "num_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 0)),
                "vocab_size": getattr(config, 'vocab_size', 0),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 0))
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None