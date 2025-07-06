#!/usr/bin/env python3
"""
Main script to run the complete diffusion conversion pipeline.
"""
import argparse
import yaml
import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.diffusion_transformer import DiffusionTransformer, DiffusionTransformerConfig
from models.noise_scheduler import MaskedDiffusionScheduler
from training.trainer import DiffusionTrainer
from training.optimizer import create_optimizer, create_scheduler
from data.dataset_loader import create_dataloaders
from utils.hardware_utils import setup_hardware, check_memory_requirements, get_system_info
from utils.model_downloader import ModelDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_configs(args):
    """Load and merge configuration files."""
    configs = {}
    
    # Load model config
    with open(args.model_config, 'r') as f:
        configs['model'] = yaml.safe_load(f)
    
    # Load training config  
    with open(args.training_config, 'r') as f:
        configs['training'] = yaml.safe_load(f)
    
    # Load hardware config
    with open(args.hardware_config, 'r') as f:
        configs['hardware'] = yaml.safe_load(f)
    
    return configs

def setup_model_and_tokenizer(model_config, hardware_config):
    """Setup the base model and tokenizer."""
    model_name = model_config['model_name']
    
    logger.info(f"Loading base model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add mask token if not present
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        model_config['mask_token_id'] = tokenizer.mask_token_id
        logger.info(f"Added mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    else:
        model_config['mask_token_id'] = tokenizer.mask_token_id
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad token to eos token: {tokenizer.pad_token}")
    
    # Load base model
    device_map = "auto" if hardware_config.get('device') != "cpu" else None
    torch_dtype = torch.float16 if hardware_config.get('dtype') == 'float16' else torch.float32
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            cache_dir=model_config.get('cache_dir', './model_cache')
        )
    except Exception as e:
        logger.warning(f"Failed to load with device_map=auto: {e}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=model_config.get('cache_dir', './model_cache')
        )
    
    # Resize embeddings if we added tokens
    if len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized embeddings to {len(tokenizer)} tokens")
    
    return base_model, tokenizer

def create_diffusion_model(base_model, model_config):
    """Create the diffusion transformer."""
    # Create diffusion config
    diffusion_config = DiffusionTransformerConfig(
        base_model_name=model_config['model_name'],
        hidden_size=model_config.get('hidden_size', base_model.config.hidden_size),
        num_timesteps=model_config.get('num_timesteps', 1000),
        mask_token_id=model_config['mask_token_id'],
        time_embedding_dim=model_config.get('time_embedding_dim'),
        use_bidirectional_attention=model_config.get('use_bidirectional_attention', True),
        freeze_base_model=model_config.get('freeze_base_model', True)
    )
    
    # Create diffusion model
    model = DiffusionTransformer(base_model, diffusion_config)
    
    logger.info(f"Created diffusion model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train diffusion language model")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config")
    parser.add_argument("--training_config", type=str, required=True, help="Path to training config")
    parser.add_argument("--hardware_config", type=str, required=True, help="Path to hardware config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry_run", action="store_true", help="Run setup without training")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configurations
    logger.info("Loading configurations...")
    configs = load_configs(args)
    model_config = configs['model']
    training_config = configs['training']
    hardware_config = configs['hardware']
    
    # Log system info
    sys_info = get_system_info()
    logger.info(f"System: {sys_info['cpu_count']} CPUs, {sys_info['memory_total_gb']:.1f}GB RAM")
    if sys_info['gpu_info']['available']:
        for gpu in sys_info['gpu_info']['devices']:
            logger.info(f"GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.1f}GB)")
    
    # Setup hardware
    logger.info("Setting up hardware...")
    device = setup_hardware(hardware_config)
    logger.info(f"Using device: {device}")
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    base_model, tokenizer = setup_model_and_tokenizer(model_config, hardware_config)
    
    # Create diffusion model
    logger.info("Creating diffusion model...")
    model = create_diffusion_model(base_model, model_config)
    
    # Check memory requirements
    if not check_memory_requirements(model, hardware_config):
        logger.error("Model may not fit in available memory!")
        if not args.dry_run:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return 1
    
    # Create noise scheduler
    logger.info("Creating noise scheduler...")
    noise_scheduler = MaskedDiffusionScheduler(
        num_timesteps=model_config.get('num_timesteps', 1000),
        schedule_type=model_config.get('noise_schedule', 'cosine'),
        mask_token_id=model_config['mask_token_id']
    )
    
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {model_config['model_name']}")
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {training_config.get('batch_size', 4)}")
        logger.info(f"Gradient accumulation: {training_config.get('gradient_accumulation_steps', 1)}")
        logger.info(f"Effective batch size: {training_config.get('batch_size', 4) * training_config.get('gradient_accumulation_steps', 1)}")
        logger.info(f"Max sequence length: {training_config.get('max_seq_length', 512)}")
        logger.info(f"Training stages: {training_config.get('stage1_epochs', 2)} + {training_config.get('stage2_epochs', 4)} + {training_config.get('stage3_epochs', 2)} epochs")
        logger.info(f"Dataset: {training_config.get('dataset_name', 'openwebtext')}")
        logger.info(f"Noise schedule: {model_config.get('noise_schedule', 'cosine')}")
        logger.info(f"Timesteps: {model_config.get('num_timesteps', 1000)}")
        logger.info(f"Bidirectional attention: {model_config.get('use_bidirectional_attention', True)}")
        logger.info("=" * 60)
        logger.info("Dry run completed successfully!")
        return 0
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader, eval_dataloader = create_dataloaders(
        tokenizer, 
        training_config
    )
    
    logger.info(f"Train batches: {len(train_dataloader)}")
    if eval_dataloader:
        logger.info(f"Eval batches: {len(eval_dataloader)}")
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(model, training_config)
    
    # Calculate training steps
    num_epochs = (
        training_config.get('stage1_epochs', 2) + 
        training_config.get('stage2_epochs', 4) + 
        training_config.get('stage3_epochs', 2)
    )
    num_training_steps = len(train_dataloader) * num_epochs
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, training_config, num_training_steps)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = DiffusionTrainer(training_config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Effective batch size: {training_config.get('batch_size', 4) * training_config.get('gradient_accumulation_steps', 1)}")
    
    try:
        model = trainer.train(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            tokenizer=tokenizer
        )
        
        logger.info("Training completed successfully!")
        
        # Save final model
        output_dir = Path(training_config.get('output_dir', './outputs'))
        final_model_dir = output_dir / "final_model"
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info(f"Final model saved to {final_model_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    sys.exit(main())