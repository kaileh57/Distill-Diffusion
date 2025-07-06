#!/usr/bin/env python3
"""
Script to download and prepare models for conversion.
"""
import argparse
import sys
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.model_downloader import ModelDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Download and analyze models for diffusion conversion")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--config", type=str, help="Path to model config (optional)")
    parser.add_argument("--cache-dir", type=str, default="./model_cache", help="Cache directory")
    parser.add_argument("--info-only", action="store_true", help="Only get model info without downloading")
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)
    
    if args.info_only:
        # Just get basic info
        print(f"Getting info for {args.model}...")
        info = downloader.get_model_info(args.model)
        if info:
            print(f"\nModel Info:")
            print(f"  Type: {info['model_type']}")
            print(f"  Hidden Size: {info['hidden_size']}")
            print(f"  Layers: {info['num_layers']}")
            print(f"  Vocab Size: {info['vocab_size']}")
            print(f"  Max Position Embeddings: {info['max_position_embeddings']}")
        else:
            print("Failed to get model info")
            return 1
    else:
        # Download and analyze
        try:
            model, tokenizer, analysis = downloader.download_model(args.model, config)
            
            print(f"\n{'='*50}")
            print(f"Model Analysis for {args.model}")
            print(f"{'='*50}")
            print(f"Total Parameters: {analysis['total_parameters_billions']:.2f}B")
            print(f"Model Type: {analysis['model_type']}")
            print(f"Layers: {analysis['num_layers']}")
            print(f"Hidden Size: {analysis['hidden_size']}")
            print(f"Attention Heads: {analysis['num_attention_heads']}")
            print(f"Vocab Size: {analysis['vocab_size']}")
            
            print(f"\nMemory Requirements:")
            print(f"  Weights (FP16): {analysis['memory_requirements']['fp16_weights_gb']:.2f} GB")
            print(f"  Weights (FP32): {analysis['memory_requirements']['fp32_weights_gb']:.2f} GB")
            print(f"  Training Estimate: {analysis['memory_requirements']['training_estimate_gb']:.2f} GB")
            print(f"  Inference Estimate: {analysis['memory_requirements']['inference_gb']:.2f} GB")
            
            print(f"\nConversion Feasibility:")
            feasibility = analysis['conversion_feasibility']
            print(f"  Can fit on single GPU: {feasibility['can_fit_single_gpu']}")
            print(f"  Requires multi-GPU: {feasibility['requires_multi_gpu']}")
            print(f"  Memory optimizations needed: {feasibility['memory_optimizations_needed']}")
            
            if feasibility['recommended_optimizations']:
                print(f"  Recommended optimizations: {', '.join(feasibility['recommended_optimizations'])}")
            
            # Hardware compatibility check
            if analysis['memory_requirements']['training_estimate_gb'] > 48:
                print(f"\n⚠️  WARNING: Model may not fit on single A6000 (48GB) for training!")
                print(f"   Consider using:")
                print(f"   - QLoRA/LoRA fine-tuning")
                print(f"   - Gradient checkpointing")
                print(f"   - Multi-GPU setup")
                print(f"   - DeepSpeed ZeRO")
            elif analysis['memory_requirements']['training_estimate_gb'] > 30:
                print(f"\n💡 TIP: Consider memory optimizations for better performance:")
                print(f"   - 8-bit Adam optimizer")
                print(f"   - Gradient checkpointing")
                print(f"   - Mixed precision training")
            else:
                print(f"\n✅ Model should fit comfortably on A6000 for training")
            
            # Save analysis
            analysis_file = Path(args.cache_dir) / f"{args.model.replace('/', '_')}_analysis.json"
            print(f"\nAnalysis saved to: {analysis_file}")
            
        except Exception as e:
            logger.error(f"Failed to download/analyze model: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())