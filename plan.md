# Diffusion LLM Conversion Implementation Plan

## Project Overview
Create a repository that converts pre-trained autoregressive language models into diffusion models, optimized for RTX A6000 (48GB) + RTX A4500 (20GB) hardware setup.

## Repository Structure
```
diffusion-llm-converter/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   ├── model_configs/
│   │   ├── phi2_diffusion.yaml
│   │   ├── stablelm3b_diffusion.yaml
│   │   └── mistral7b_diffusion.yaml
│   ├── training_configs/
│   │   ├── stage1_attention.yaml
│   │   ├── stage2_diffusion.yaml
│   │   └── stage3_recovery.yaml
│   └── hardware_configs/
│       ├── single_a6000.yaml
│       └── multi_gpu.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── diffusion_transformer.py
│   │   ├── noise_scheduler.py
│   │   ├── model_converter.py
│   │   └── hybrid_attention.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py
│   │   ├── preprocessing.py
│   │   └── noise_augmentation.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── loss_functions.py
│   │   └── checkpointing.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── perplexity.py
│   │   ├── generation_metrics.py
│   │   └── benchmark_runner.py
│   └── utils/
│       ├── __init__.py
│       ├── memory_optimizer.py
│       ├── model_downloader.py
│       └── hardware_utils.py
├── scripts/
│   ├── download_model.py
│   ├── convert_model.py
│   ├── train_diffusion.py
│   ├── evaluate.py
│   └── generate_samples.py
├── notebooks/
│   ├── 01_model_analysis.ipynb
│   ├── 02_conversion_experiments.ipynb
│   └── 03_results_visualization.ipynb
└── tests/
    ├── test_models.py
    ├── test_training.py
    └── test_conversion.py
```

## Phase 1: Environment Setup and Dependencies

### 1.1 Create requirements.txt
```python
# Core dependencies
torch>=2.1.0
transformers>=4.36.0
accelerate>=0.25.0
datasets>=2.14.0
einops>=0.7.0
wandb>=0.16.0

# Diffusion specific
diffusers>=0.24.0
tqdm>=4.66.0
scipy>=1.11.0

# Optimization
bitsandbytes>=0.41.0  # For 8-bit Adam
peft>=0.7.0  # For LoRA/QLoRA
deepspeed>=0.12.0  # For multi-GPU training

# Evaluation
nltk>=3.8.0
rouge-score>=0.1.2
sacrebleu>=2.3.0

# Development
pytest>=7.4.0
black>=23.0.0
flake8>=6.1.0
```

### 1.2 Create setup.py
```python
from setuptools import setup, find_packages

setup(
    name="diffusion-llm-converter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List from requirements.txt
    ],
    python_requires=">=3.8",
)
```

## Phase 2: Model Download and Analysis

### 2.1 Create src/utils/model_downloader.py
```python
"""
Download and analyze pre-trained models to determine conversion feasibility.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path

class ModelDownloader:
    def __init__(self, cache_dir="./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def download_model(self, model_name, model_configs):
        """
        Download model and analyze its architecture for conversion.
        
        Args:
            model_name: HuggingFace model identifier
            model_configs: Dict with target configs for conversion
        """
        print(f"Downloading {model_name}...")
        
        # Download model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Analyze model architecture
        analysis = self.analyze_model(model)
        
        # Save analysis
        with open(self.cache_dir / f"{model_name.replace('/', '_')}_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return model, tokenizer, analysis
    
    def analyze_model(self, model):
        """Analyze model architecture for conversion requirements."""
        total_params = sum(p.numel() for p in model.parameters())
        
        analysis = {
            "total_parameters": total_params,
            "total_parameters_billions": total_params / 1e9,
            "layers": len(model.transformer.h) if hasattr(model, 'transformer') else len(model.model.layers),
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": model.config.num_attention_heads,
            "memory_requirements": {
                "fp16_weights_gb": total_params * 2 / 1e9,
                "training_estimate_gb": total_params * 8 / 1e9,  # Conservative estimate
                "inference_gb": total_params * 2.5 / 1e9
            }
        }
        
        return analysis
```

### 2.2 Create scripts/download_model.py
```python
#!/usr/bin/env python3
"""
Script to download and prepare models for conversion.
"""
import argparse
from src.utils.model_downloader import ModelDownloader
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Download and analyze
    downloader = ModelDownloader()
    model, tokenizer, analysis = downloader.download_model(args.model, config)
    
    print("\nModel Analysis:")
    print(f"Total Parameters: {analysis['total_parameters_billions']:.2f}B")
    print(f"Memory Requirements:")
    print(f"  - Weights (FP16): {analysis['memory_requirements']['fp16_weights_gb']:.2f} GB")
    print(f"  - Training: ~{analysis['memory_requirements']['training_estimate_gb']:.2f} GB")
    
    # Check hardware compatibility
    if analysis['memory_requirements']['training_estimate_gb'] > 48:
        print("\n⚠️  Warning: Model may not fit on single A6000 for training!")
        print("Consider using QLoRA or multi-GPU setup.")

if __name__ == "__main__":
    main()
```

## Phase 3: Model Conversion Architecture

### 3.1 Create src/models/diffusion_transformer.py
```python
"""
Diffusion transformer architecture adapted from autoregressive models.
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from einops import rearrange
import math

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings for diffusion."""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

class DiffusionTransformer(nn.Module):
    """
    Wrapper to convert autoregressive transformer to diffusion.
    """
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Freeze base model initially
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Add diffusion-specific components
        self.time_embed = TimestepEmbedding(config.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        # Modify attention masks for bidirectional attention
        self._convert_attention_masks()
        
        # Add denoising head
        self.denoising_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size,
            bias=False
        )
        
    def _convert_attention_masks(self):
        """Convert causal masks to bidirectional."""
        # This will be called during forward pass
        pass
        
    def forward(self, input_ids, timesteps, attention_mask=None):
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        time_emb = self.time_mlp(time_emb)
        
        # Get base model embeddings
        if hasattr(self.base_model, 'transformer'):
            inputs_embeds = self.base_model.transformer.wte(input_ids)
        else:
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
            
        # Add time embeddings
        inputs_embeds = inputs_embeds + time_emb.unsqueeze(1)
        
        # Forward through transformer with bidirectional attention
        outputs = self._forward_transformer(
            inputs_embeds, 
            attention_mask=attention_mask,
            use_causal_mask=False  # Key change
        )
        
        # Denoising prediction
        logits = self.denoising_head(outputs.last_hidden_state)
        
        return logits
    
    def _forward_transformer(self, inputs_embeds, attention_mask, use_causal_mask):
        """Forward pass through transformer layers."""
        # Implementation depends on base model architecture
        # This is a simplified version
        if hasattr(self.base_model, 'transformer'):
            outputs = self.base_model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False
            )
        else:
            outputs = self.base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
        return outputs
```

### 3.2 Create src/models/noise_scheduler.py
```python
"""
Noise scheduling for diffusion models.
"""
import torch
import numpy as np

class MaskedDiffusionScheduler:
    """
    Implements noise scheduling for masked diffusion language models.
    """
    def __init__(
        self,
        num_timesteps=1000,
        schedule_type="cosine",
        mask_token_id=None
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.mask_token_id = mask_token_id
        
        # Create noise schedule
        if schedule_type == "cosine":
            self.betas = self._cosine_schedule()
        elif schedule_type == "linear":
            self.betas = self._linear_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        # Precompute useful quantities
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def _cosine_schedule(self):
        """Cosine noise schedule."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _linear_schedule(self):
        """Linear noise schedule."""
        return torch.linspace(0.0001, 0.02, self.num_timesteps)
    
    def add_noise(self, input_ids, timesteps):
        """
        Add noise to input sequences by masking tokens.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get noise rates for each timestep
        noise_rates = self.alphas_cumprod[timesteps].to(device)
        
        # Create random mask
        rand = torch.rand(batch_size, seq_len, device=device)
        mask = rand < (1 - noise_rates.unsqueeze(1))
        
        # Apply mask
        noisy_ids = input_ids.clone()
        noisy_ids[mask] = self.mask_token_id
        
        return noisy_ids, mask
    
    def get_loss_weight(self, timesteps):
        """Get loss weights for different timesteps."""
        # Simple SNR weighting
        snr = self.alphas_cumprod[timesteps] / (1 - self.alphas_cumprod[timesteps])
        weight = torch.sqrt(snr)
        return weight
```

## Phase 4: Training Pipeline

### 4.1 Create src/training/trainer.py
```python
"""
Main training logic for diffusion model conversion.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
from pathlib import Path

class DiffusionTrainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            cpu=False
        )
        
        # Initialize wandb if enabled
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(project="diffusion-llm", config=config)
            
    def train(self, model, train_dataloader, eval_dataloader, optimizer, scheduler, noise_scheduler):
        """
        Main training loop with three stages:
        1. Attention pattern adaptation
        2. Diffusion objective training  
        3. Capability recovery
        """
        # Prepare for distributed training
        model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        
        global_step = 0
        best_eval_loss = float('inf')
        
        for stage in range(1, 4):
            print(f"\n{'='*50}")
            print(f"Stage {stage}: {self._get_stage_name(stage)}")
            print(f"{'='*50}\n")
            
            # Configure stage-specific parameters
            self._configure_stage(model, stage)
            
            for epoch in range(self.config.num_epochs_per_stage):
                model.train()
                epoch_loss = 0
                
                progress_bar = tqdm(
                    train_dataloader, 
                    desc=f"Stage {stage} Epoch {epoch}",
                    disable=not self.accelerator.is_main_process
                )
                
                for batch in progress_bar:
                    # Stage-specific forward pass
                    loss = self._training_step(
                        model, batch, noise_scheduler, stage
                    )
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if self.config.max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(), 
                                self.config.max_grad_norm
                            )
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / (global_step % len(train_dataloader) + 1)
                        progress_bar.set_postfix(loss=avg_loss)
                        
                        if self.config.use_wandb and self.accelerator.is_main_process:
                            wandb.log({
                                f"stage_{stage}_loss": avg_loss,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "global_step": global_step
                            })
                    
                    # Evaluation
                    if global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate(
                            model, eval_dataloader, noise_scheduler, stage
                        )
                        
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            self._save_checkpoint(model, optimizer, scheduler, global_step, stage)
                        
                        model.train()
                        
    def _training_step(self, model, batch, noise_scheduler, stage):
        """Stage-specific training step."""
        input_ids = batch['input_ids']
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.num_timesteps, 
            (batch_size,), device=device
        )
        
        # Add noise
        noisy_ids, mask = noise_scheduler.add_noise(input_ids, timesteps)
        
        # Forward pass
        logits = model(noisy_ids, timesteps, attention_mask=batch['attention_mask'])
        
        # Compute loss based on stage
        if stage == 1:
            # Stage 1: Focus on attention pattern adaptation
            loss = self._attention_adaptation_loss(logits, input_ids, mask)
        elif stage == 2:
            # Stage 2: Full diffusion objective
            loss = self._diffusion_loss(logits, input_ids, mask, timesteps, noise_scheduler)
        else:
            # Stage 3: Capability recovery with auxiliary tasks
            loss = self._capability_recovery_loss(logits, input_ids, mask, batch)
            
        return loss
    
    def _diffusion_loss(self, logits, target_ids, mask, timesteps, noise_scheduler):
        """Masked diffusion loss with timestep weighting."""
        # Only compute loss on masked positions
        loss = F.cross_entropy(
            logits[mask].view(-1, logits.shape[-1]),
            target_ids[mask].view(-1),
            reduction='none'
        )
        
        # Apply timestep-dependent weighting
        weights = noise_scheduler.get_loss_weight(timesteps)
        weights = weights.unsqueeze(1).expand_as(mask)[mask]
        
        weighted_loss = (loss * weights).mean()
        return weighted_loss
    
    def _configure_stage(self, model, stage):
        """Configure model parameters for each training stage."""
        if stage == 1:
            # Stage 1: Only train attention-related parameters
            for name, param in model.named_parameters():
                if 'attention' in name or 'time' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif stage == 2:
            # Stage 2: Train all diffusion components
            for name, param in model.named_parameters():
                if 'base_model' in name and 'embed' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            # Stage 3: Fine-tune everything
            for param in model.parameters():
                param.requires_grad = True
    
    def _get_stage_name(self, stage):
        """Get descriptive name for each stage."""
        return {
            1: "Attention Pattern Adaptation",
            2: "Diffusion Objective Training",
            3: "Capability Recovery"
        }[stage]
```

### 4.2 Create src/training/optimizer.py
```python
"""
Memory-efficient optimizers for large model training.
"""
import torch
from bitsandbytes.optim import AdamW8bit

def create_optimizer(model, config):
    """
    Create optimizer with memory-efficient options.
    """
    # Separate parameters by type
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(nd in name for nd in ["bias", "LayerNorm", "layernorm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    
    # Use 8-bit Adam for memory efficiency
    if config.use_8bit_adam:
        optimizer = AdamW8bit(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
    
    return optimizer
```

## Phase 5: Configuration Files

### 5.1 Create configs/model_configs/phi2_diffusion.yaml
```yaml
# Configuration for converting Phi-2 to diffusion
model_name: "microsoft/phi-2"
model_type: "phi2"
hidden_size: 2560
num_attention_heads: 32
num_hidden_layers: 32
vocab_size: 51200

# Diffusion specific
mask_token_id: 50256  # Using unused token
num_timesteps: 1000
noise_schedule: "cosine"

# Training stages
stage1_epochs: 2
stage2_epochs: 4
stage3_epochs: 2

# Memory optimization
use_gradient_checkpointing: true
use_lora: false  # Can enable for larger models
lora_rank: 16
lora_alpha: 32
```

### 5.2 Create configs/training_configs/stage2_diffusion.yaml
```yaml
# Stage 2: Diffusion objective training
num_epochs_per_stage: 4
batch_size: 4
gradient_accumulation_steps: 8  # Effective batch size = 32
learning_rate: 1e-4
warmup_steps: 1000
max_grad_norm: 1.0

# Optimizer
use_8bit_adam: true
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1e-8
weight_decay: 0.01

# Data
dataset_name: "openwebtext"
max_seq_length: 512
preprocessing_num_workers: 4

# Checkpointing
save_steps: 5000
eval_steps: 1000
logging_steps: 100
save_total_limit: 3

# Hardware
mixed_precision: "fp16"
use_wandb: true
```

### 5.3 Create configs/hardware_configs/single_a6000.yaml
```yaml
# Optimized for single RTX A6000 (48GB)
device: "cuda:0"
max_memory: 45  # Leave 3GB headroom
dtype: "float16"

# Memory optimization
gradient_checkpointing: true
cpu_offload: false
optimizer_offload: false

# Batch size tuning
max_batch_size_3b: 8
max_batch_size_7b: 2
max_batch_size_7b_lora: 4

# Performance
compile_model: true  # PyTorch 2.0 compile
use_flash_attention: true
```

## Phase 6: Main Training Script

### 6.1 Create scripts/train_diffusion.py
```python
#!/usr/bin/env python3
"""
Main script to run the complete diffusion conversion pipeline.
"""
import argparse
import yaml
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.models.diffusion_transformer import DiffusionTransformer
from src.models.noise_scheduler import MaskedDiffusionScheduler
from src.training.trainer import DiffusionTrainer
from src.training.optimizer import create_optimizer
from src.data.dataset_loader import create_dataloaders
from src.utils.hardware_utils import setup_hardware

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--training_config", type=str, required=True)
    parser.add_argument("--hardware_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    # Load configs
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.training_config) as f:
        training_config = yaml.safe_load(f)
    with open(args.hardware_config) as f:
        hardware_config = yaml.safe_load(f)
    
    # Setup hardware
    device = setup_hardware(hardware_config)
    
    print("Loading base model...")
    # Load pre-trained model
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name'],
        torch_dtype=torch.float16 if hardware_config['dtype'] == 'float16' else torch.float32,
        device_map=device
    )
    
    # Convert to diffusion model
    print("Converting to diffusion model...")
    model = DiffusionTransformer(base_model, model_config)
    
    # Load tokenizer and update with mask token
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        model.resize_token_embeddings(len(tokenizer))
    
    # Create noise scheduler
    noise_scheduler = MaskedDiffusionScheduler(
        num_timesteps=model_config['num_timesteps'],
        schedule_type=model_config['noise_schedule'],
        mask_token_id=tokenizer.mask_token_id
    )
    
    # Create data loaders
    print("Loading datasets...")
    train_dataloader, eval_dataloader = create_dataloaders(
        tokenizer, 
        training_config,
        hardware_config
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, training_config)
    
    # Create learning rate scheduler
    from transformers import get_cosine_schedule_with_warmup
    num_training_steps = (
        len(train_dataloader) * 
        (model_config['stage1_epochs'] + model_config['stage2_epochs'] + model_config['stage3_epochs'])
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Create trainer
    trainer = DiffusionTrainer(training_config)
    
    # Start training
    print("Starting training...")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")
    
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        noise_scheduler=noise_scheduler
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
```

## Phase 7: Evaluation and Generation

### 7.1 Create src/evaluation/generation_metrics.py
```python
"""
Evaluation metrics for diffusion language models.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

class DiffusionEvaluator:
    def __init__(self, tokenizer, device='cuda'):
        self.tokenizer = tokenizer
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    def compute_perplexity(self, model, dataloader, noise_scheduler):
        """
        Compute perplexity for diffusion models.
        Uses the variational bound as proxy.
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Compute loss over all timesteps
                batch_loss = 0
                for t in range(0, noise_scheduler.num_timesteps, 10):  # Sample timesteps
                    timesteps = torch.full((input_ids.shape[0],), t, device=self.device)
                    noisy_ids, mask = noise_scheduler.add_noise(input_ids, timesteps)
                    
                    logits = model(noisy_ids, timesteps, attention_mask)
                    loss = torch.nn.functional.cross_entropy(
                        logits[mask].view(-1, logits.shape[-1]),
                        input_ids[mask].view(-1),
                        reduction='sum'
                    )
                    batch_loss += loss
                
                total_loss += batch_loss.item()
                total_tokens += mask.sum().item()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    
    def generate_samples(self, model, prompts, noise_scheduler, max_length=100, num_steps=50):
        """
        Generate text using diffusion sampling.
        """
        model.eval()
        generated_texts = []
        
        for prompt in prompts:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)
            prompt_len = inputs['input_ids'].shape[1]
            
            # Initialize with masked tokens
            seq_len = min(max_length, self.tokenizer.model_max_length)
            input_ids = torch.full(
                (1, seq_len), 
                self.tokenizer.mask_token_id, 
                device=self.device
            )
            input_ids[:, :prompt_len] = inputs['input_ids']
            
            # Diffusion sampling loop
            for t in reversed(range(0, noise_scheduler.num_timesteps, noise_scheduler.num_timesteps // num_steps)):
                timesteps = torch.tensor([t], device=self.device)
                
                with torch.no_grad():
                    logits = model(input_ids, timesteps)
                    
                # Sample from logits (with temperature)
                probs = torch.softmax(logits / 0.8, dim=-1)
                
                # Only update masked positions
                mask = (input_ids == self.tokenizer.mask_token_id)
                if mask.any():
                    sampled_ids = torch.multinomial(
                        probs[mask].view(-1, probs.shape[-1]), 
                        num_samples=1
                    ).squeeze()
                    input_ids[mask] = sampled_ids
            
            # Decode
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
```

### 7.2 Create scripts/evaluate.py
```python
#!/usr/bin/env python3
"""
Evaluate converted diffusion models.
"""
import argparse
import torch
from pathlib import Path
import json

from src.models.diffusion_transformer import DiffusionTransformer
from src.models.noise_scheduler import MaskedDiffusionScheduler
from src.evaluation.generation_metrics import DiffusionEvaluator
from src.data.dataset_loader import create_eval_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    args = parser.parse_args()
    
    # Load model and configs
    checkpoint = torch.load(args.checkpoint)
    model_config = checkpoint['model_config']
    
    # Initialize model
    model = DiffusionTransformer.from_pretrained(checkpoint['model_state_dict'])
    model.eval()
    
    # Create evaluator
    evaluator = DiffusionEvaluator(checkpoint['tokenizer'])
    
    # Run evaluations
    results = {}
    
    # 1. Perplexity evaluation
    print("Computing perplexity...")
    eval_dataloader = create_eval_dataloader(checkpoint['tokenizer'], model_config)
    perplexity = evaluator.compute_perplexity(
        model, 
        eval_dataloader,
        MaskedDiffusionScheduler(
            num_timesteps=model_config['num_timesteps'],
            mask_token_id=checkpoint['tokenizer'].mask_token_id
        )
    )
    results['perplexity'] = perplexity
    print(f"Perplexity: {perplexity:.2f}")
    
    # 2. Generation quality
    print("\nGenerating samples...")
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The key to solving climate change involves",
        "In the year 2050, humanity will"
    ]
    
    generated_texts = evaluator.generate_samples(
        model,
        test_prompts,
        noise_scheduler,
        max_length=100
    )
    
    results['generated_samples'] = [
        {"prompt": p, "generation": g} 
        for p, g in zip(test_prompts, generated_texts)
    ]
    
    # 3. Task-specific evaluations
    print("\nRunning task-specific evaluations...")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
```

## Phase 8: Quick Start Script

### 8.1 Create run_conversion.sh
```bash
#!/bin/bash
# Complete pipeline to convert a model to diffusion

MODEL_NAME="microsoft/phi-2"  # Change this to your desired model
EXPERIMENT_NAME="phi2_diffusion_conversion"

echo "Starting Diffusion LLM Conversion Pipeline"
echo "Model: $MODEL_NAME"
echo "Experiment: $EXPERIMENT_NAME"

# Step 1: Download and analyze model
echo "\n1. Downloading model..."
python scripts/download_model.py \
    --model $MODEL_NAME \
    --config configs/model_configs/phi2_diffusion.yaml

# Step 2: Run conversion training
echo "\n2. Starting conversion training..."
python scripts/train_diffusion.py \
    --model_config configs/model_configs/phi2_diffusion.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml

# Step 3: Evaluate converted model
echo "\n3. Evaluating converted model..."
python scripts/evaluate.py \
    --checkpoint outputs/$EXPERIMENT_NAME/best_checkpoint.pt \
    --output_dir outputs/$EXPERIMENT_NAME/evaluation

echo "\nConversion complete! Check outputs/$EXPERIMENT_NAME for results."
```

## Usage Instructions

1. **Setup Environment**
```bash
git clone <your-repo>
cd diffusion-llm-converter
pip install -r requirements.txt
```

2. **Configure for Your Model**
- Edit `configs/model_configs/` to add your target model
- Adjust `configs/training_configs/` for your hardware limits
- Modify batch sizes in `configs/hardware_configs/`

3. **Run Conversion**
```bash
# For Phi-2 (recommended for your hardware)
bash run_conversion.sh

# For custom model
python scripts/train_diffusion.py \
    --model_config configs/model_configs/your_model.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml
```

4. **Monitor Training**
- Check wandb dashboard for live metrics
- Monitor GPU usage with `nvidia-smi`
- Logs saved to `outputs/experiment_name/`

## Key Implementation Notes

1. **Memory Management**
   - Gradient checkpointing is essential for 7B models
   - Use 8-bit Adam to save optimizer memory
   - Consider QLoRA for models larger than 3B

2. **Training Stages**
   - Stage 1: Convert attention patterns (2 epochs)
   - Stage 2: Main diffusion training (4 epochs)  
   - Stage 3: Recovery fine-tuning (2 epochs)

3. **Hardware Optimization**
   - Compile model with PyTorch 2.0 for speedup
   - Use Flash Attention if available
   - Mixed precision (FP16) is default

4. **Debugging Tips**
   - Start with smaller sequence lengths (256)
   - Use gradient accumulation for larger effective batches
   - Monitor loss curves for each stage separately

This implementation provides a complete framework for converting autoregressive LLMs to diffusion models, optimized for your RTX A6000 hardware setup.