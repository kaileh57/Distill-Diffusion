# Diffusion LLM Converter

Convert pre-trained autoregressive language models into diffusion models, optimized for RTX A6000 (48GB) + RTX A4500 (20GB) hardware setup.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo>
cd Distill-Diffusion

# Install dependencies
pip install -r requirements.txt

# Run the complete conversion pipeline
./run_conversion.sh
```

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Hardware Requirements](#hardware-requirements)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Stages](#training-stages)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🔍 Overview

This project implements a complete pipeline for converting autoregressive language models (like GPT-2, Phi-2, LLaMA) into diffusion models. The conversion process uses a three-stage training approach:

1. **Attention Pattern Adaptation**: Adapt the model's attention patterns for bidirectional processing
2. **Diffusion Objective Training**: Train on the full diffusion objective with masked language modeling
3. **Capability Recovery**: Fine-tune to recover original language modeling capabilities

### Key Features

- 🎯 **Three-stage training** for stable conversion
- 🔧 **Memory-efficient** training with 8-bit optimizers and gradient checkpointing  
- 🎨 **Flexible configuration** system for different models and hardware
- 📊 **Comprehensive monitoring** with Weights & Biases integration
- 🔄 **Resume training** from checkpoints
- 💾 **Automatic memory management** and optimization

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: RTX A6000 48GB)
- 32GB+ RAM recommended

### Quick Install

```bash
# Install from source
git clone <your-repo>
cd Distill-Diffusion
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Manual Dependencies

```bash
pip install torch>=2.1.0 transformers>=4.36.0 accelerate>=0.25.0
pip install datasets einops wandb diffusers tqdm scipy
pip install bitsandbytes peft deepspeed  # For memory optimization
pip install nltk rouge-score sacrebleu    # For evaluation
```

## 💻 Hardware Requirements

### Minimum Requirements
- **GPU**: 24GB VRAM (RTX 3090/4090, A5000)
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space

### Recommended Setup (Target Hardware)
- **Primary GPU**: RTX A6000 (48GB)
- **Secondary GPU**: RTX A4500 (20GB) - for multi-GPU training
- **RAM**: 64GB+ system RAM
- **Storage**: 200GB+ NVMe SSD

### Model Size Guidelines

| Model Size | Min VRAM | Recommended VRAM | Batch Size | Notes |
|------------|----------|------------------|------------|-------|
| 1B params  | 16GB     | 24GB            | 4-8        | GPT-2 Large |
| 3B params  | 24GB     | 48GB            | 2-4        | Phi-2 |
| 7B params  | 48GB     | 80GB            | 1-2        | LLaMA-7B |
| 13B params | 80GB     | 160GB           | 1          | Requires multi-GPU |

## 🎯 Usage

### Basic Usage

```bash
# Quick start with Phi-2
./run_conversion.sh

# Custom model
python scripts/train_diffusion.py \
    --model_config configs/model_configs/your_model.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml
```

### Step-by-Step Usage

#### 1. Download and Analyze Model

```bash
python scripts/download_model.py \
    --model microsoft/phi-2 \
    --config configs/model_configs/phi2_diffusion.yaml
```

#### 2. Validate Setup (Dry Run)

```bash
python scripts/train_diffusion.py \
    --model_config configs/model_configs/phi2_diffusion.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml \
    --dry_run
```

#### 3. Start Training

```bash
python scripts/train_diffusion.py \
    --model_config configs/model_configs/phi2_diffusion.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml
```

#### 4. Resume from Checkpoint

```bash
python scripts/train_diffusion.py \
    --model_config configs/model_configs/phi2_diffusion.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml \
    --checkpoint outputs/phi2_diffusion/checkpoint_10000
```

## ⚙️ Configuration

### Model Configuration (`configs/model_configs/`)

```yaml
# Example: phi2_diffusion.yaml
model_name: "microsoft/phi-2"
model_type: "phi"
hidden_size: 2560
num_timesteps: 1000
noise_schedule: "cosine"
use_bidirectional_attention: true

# Training stages
stage1_epochs: 2  # Attention adaptation
stage2_epochs: 4  # Diffusion training  
stage3_epochs: 2  # Capability recovery

# Memory optimization
use_gradient_checkpointing: true
use_lora: false
```

### Training Configuration (`configs/training_configs/`)

```yaml
# Example: stage2_diffusion.yaml
batch_size: 2
gradient_accumulation_steps: 16  # Effective batch size = 32
learning_rate: 1e-4
use_8bit_adam: true

# Data
dataset_name: "wikitext"
max_seq_length: 512
max_train_samples: 50000

# Monitoring
use_wandb: true
save_steps: 2500
eval_steps: 1000
```

### Hardware Configuration (`configs/hardware_configs/`)

```yaml
# Example: single_a6000.yaml
device: "cuda:0"
max_memory_gb: 45  # Leave 3GB headroom
dtype: "float16"

# Optimizations
gradient_checkpointing: true
use_8bit_adam: true
compile_model: true
```

## 🎓 Training Stages

### Stage 1: Attention Pattern Adaptation (2 epochs)
- **Purpose**: Adapt causal attention to bidirectional attention
- **Trainable**: Attention layers + time embeddings only
- **Learning Rate**: 0.5x base rate
- **Focus**: Attention mechanism modification

### Stage 2: Diffusion Objective Training (4 epochs)  
- **Purpose**: Train on full diffusion objective
- **Trainable**: All diffusion components
- **Learning Rate**: 1.0x base rate
- **Focus**: Masked language modeling with timestep conditioning

### Stage 3: Capability Recovery (2 epochs)
- **Purpose**: Recover original language modeling capabilities
- **Trainable**: All parameters
- **Learning Rate**: 0.3x base rate  
- **Focus**: Fine-tuning + auxiliary language modeling loss

## 📊 Monitoring

### Weights & Biases Integration

```python
# Automatic logging enabled with use_wandb: true
# Monitor at: https://wandb.ai/your-project/diffusion-llm-conversion
```

### Key Metrics to Monitor

- **Training Loss**: Should decrease steadily in each stage
- **Evaluation Loss**: Should not increase significantly  
- **Memory Usage**: Should stay below 90% of available VRAM
- **Gradient Norm**: Should remain stable (< 1.0 with clipping)

### Local Monitoring

```bash
# Watch training logs
tail -f outputs/experiment_name/training_logs.json

# Monitor GPU usage
nvidia-smi -l 1

# Check disk usage
du -sh model_cache/ outputs/
```

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory (OOM)

**Symptoms**: CUDA out of memory errors

**Solutions**:
```yaml
# Reduce batch size
batch_size: 1
gradient_accumulation_steps: 32

# Enable memory optimizations
use_gradient_checkpointing: true
use_8bit_adam: true
cpu_offload: true
```

#### Slow Training

**Symptoms**: Very slow iterations per second

**Solutions**:
```yaml
# Enable optimizations
compile_model: true
use_flash_attention: true  # If available
pin_memory: true

# Reduce sequence length
max_seq_length: 256
```

#### Loss Not Decreasing

**Symptoms**: Training loss plateaus or increases

**Solutions**:
- Check learning rate (try 5e-5 to 2e-4)
- Verify data quality and preprocessing
- Ensure mask_token_id is correct
- Check gradient clipping (max_grad_norm: 1.0)

#### Import Errors

**Symptoms**: ModuleNotFoundError

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"

# Install additional dependencies
pip install flash-attn --no-build-isolation  # Optional
```

### Debug Mode

```bash
python scripts/train_diffusion.py \
    --model_config configs/model_configs/phi2_diffusion.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml \
    --debug
```

## 📁 Project Structure

```
Distill-Diffusion/
├── configs/
│   ├── model_configs/          # Model-specific configurations
│   ├── training_configs/       # Training hyperparameters
│   └── hardware_configs/       # Hardware optimizations
├── src/
│   ├── models/                 # Core model implementations
│   ├── training/               # Training logic and optimizers
│   ├── data/                   # Data loading utilities
│   ├── evaluation/             # Evaluation metrics
│   └── utils/                  # Utility functions
├── scripts/                    # Main execution scripts
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests
└── outputs/                    # Training outputs and checkpoints
```

## 🚀 Advanced Usage

### Multi-GPU Training

```yaml
# hardware_config.yaml
use_deepspeed: true
deepspeed_config: "configs/deepspeed_zero2.json"
```

### Custom Models

1. Create model config in `configs/model_configs/`
2. Test with dry run
3. Adjust batch size and memory settings
4. Start training

### Custom Datasets

```yaml
# training_config.yaml
dataset_name: "your_dataset"
text_column: "content" 
cache_dir: "./custom_cache"
```

## 🎯 Performance Tips

### Memory Optimization
- Use `use_8bit_adam: true` (saves ~50% optimizer memory)
- Enable `gradient_checkpointing: true` (trades compute for memory)
- Reduce `max_seq_length` if possible
- Use `batch_size: 1` with high `gradient_accumulation_steps`

### Speed Optimization  
- Set `compile_model: true` (PyTorch 2.0+)
- Use `pin_memory: true` and `non_blocking: true`
- Increase `dataloader_num_workers` if CPU allows
- Use local SSD for data caching

### Quality Optimization
- Use larger datasets when possible
- Experiment with different noise schedules
- Tune learning rates per stage
- Monitor evaluation metrics carefully

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{diffusion_llm_converter,
  title={Diffusion LLM Converter: Converting Autoregressive Models to Diffusion Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/diffusion-llm-converter}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Open an issue with the "enhancement" label  
- 📖 **Documentation**: Check the wiki or open an issue
- 💬 **Questions**: Use GitHub Discussions

## 🌟 Acknowledgments

- Hugging Face Transformers for model implementations
- The diffusion models research community
- PyTorch and CUDA teams for optimization tools

---

**Happy Diffusing! 🌊**