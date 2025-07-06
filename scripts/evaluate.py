#!/usr/bin/env python3
"""
Evaluate converted diffusion models.
"""
import argparse
import torch
import sys
import json
import yaml
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.diffusion_transformer import DiffusionTransformer
from models.noise_scheduler import MaskedDiffusionScheduler
from evaluation.generation_metrics import DiffusionEvaluator
from data.dataset_loader import create_eval_dataloader
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_config(checkpoint_path: str):
    """Load model and configuration from checkpoint."""
    checkpoint_dir = Path(checkpoint_path)
    
    # Load training state to get config
    training_state_path = checkpoint_dir / 'training_state.pt'
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location='cpu')
        config = training_state.get('config', {})
    else:
        config = {}
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {checkpoint_dir}: {e}")
        return None, None, None
    
    # Load model
    try:
        model = DiffusionTransformer.from_pretrained(checkpoint_dir)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_dir}: {e}")
        return None, None, None
    
    return model, tokenizer, config

def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion language model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_eval_batches", type=int, default=100, help="Max batches for perplexity")
    parser.add_argument("--num_generation_samples", type=int, default=10, help="Number of generation samples")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation only")
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model and config
    logger.info(f"Loading model from {args.checkpoint}")
    model, tokenizer, config = load_model_and_config(args.checkpoint)
    
    if model is None:
        logger.error("Failed to load model")
        return 1
    
    # Move model to device
    model = model.to(args.device)
    logger.info(f"Model loaded on {args.device}")
    
    # Create noise scheduler
    model_config = config.get('model', config)  # Handle both nested and flat config
    noise_scheduler = MaskedDiffusionScheduler(
        num_timesteps=model_config.get('num_timesteps', 1000),
        schedule_type=model_config.get('noise_schedule', 'cosine'),
        mask_token_id=tokenizer.mask_token_id
    )
    
    # Create evaluator
    evaluator = DiffusionEvaluator(tokenizer, args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        "checkpoint_path": str(args.checkpoint),
        "model_config": model_config,
        "evaluation_timestamp": str(torch.datetime.datetime.now()),
        "device": args.device
    }
    
    if args.quick:
        # Quick evaluation only
        logger.info("Running quick evaluation...")
        quick_results = evaluator.quick_evaluation(model, noise_scheduler, args.num_generation_samples)
        results["quick_evaluation"] = quick_results
        
        # Print sample results
        logger.info("Sample generations:")
        for i, sample in enumerate(quick_results["generated_samples"][:3]):
            logger.info(f"  {i+1}. Prompt: {sample['prompt']}")
            logger.info(f"     Generated: {sample['generation']}")
        
    else:
        # Full evaluation
        logger.info("Running full evaluation...")
        
        # 1. Perplexity evaluation
        logger.info("Computing perplexity...")
        try:
            eval_dataloader = create_eval_dataloader(tokenizer, config.get('training', config))
            if eval_dataloader is not None:
                perplexity = evaluator.compute_perplexity(
                    model, eval_dataloader, noise_scheduler, max_batches=args.max_eval_batches
                )
                results['perplexity'] = float(perplexity)
                logger.info(f"Perplexity: {perplexity:.2f}")
            else:
                logger.warning("Could not create evaluation dataloader")
                results['perplexity'] = None
        except Exception as e:
            logger.error(f"Perplexity computation failed: {e}")
            results['perplexity'] = None
        
        # 2. Generation quality evaluation
        logger.info("Evaluating generation quality...")
        try:
            generation_metrics = evaluator.compute_generation_metrics(
                model, noise_scheduler
            )
            results['generation_metrics'] = generation_metrics
            
            logger.info(f"Average generation length: {generation_metrics['avg_generation_length']:.1f} tokens")
            
            # Print some sample generations
            logger.info("Sample generations:")
            for i, sample in enumerate(generation_metrics["generated_samples"][:3]):
                logger.info(f"  {i+1}. Prompt: {sample['prompt']}")
                logger.info(f"     Generated: {sample['generation'][:100]}...")
                
        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}")
            results['generation_metrics'] = None
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Save a summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Diffusion Model Evaluation Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Timestamp: {results.get('evaluation_timestamp', 'Unknown')}\n\n")
        
        if 'perplexity' in results and results['perplexity'] is not None:
            f.write(f"Perplexity: {results['perplexity']:.2f}\n")
        
        if 'generation_metrics' in results and results['generation_metrics'] is not None:
            gm = results['generation_metrics']
            f.write(f"Average generation length: {gm['avg_generation_length']:.1f} tokens\n")
            f.write(f"Generation samples: {gm['num_prompts']}\n")
        
        if 'quick_evaluation' in results:
            qe = results['quick_evaluation']
            f.write(f"Quick evaluation samples: {qe['num_prompts']}\n")
            f.write(f"Average generation length: {qe['avg_generation_length']:.1f} tokens\n")
    
    logger.info(f"Summary saved to {summary_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())