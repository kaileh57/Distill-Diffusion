#!/bin/bash
# Complete pipeline to convert a model to diffusion

# Configuration
MODEL_NAME="microsoft/phi-2"  # Change this to your desired model
EXPERIMENT_NAME="phi2_diffusion_conversion"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Diffusion LLM Conversion Pipeline${NC}"
echo -e "${BLUE}Model: $MODEL_NAME${NC}"
echo -e "${BLUE}Experiment: $EXPERIMENT_NAME${NC}"
echo ""

# Check if Python and pip are available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed${NC}"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo -e "${RED}❌ pip is not installed${NC}"
    exit 1
fi

# Check if CUDA is available (optional)
echo -e "${YELLOW}🔍 Checking CUDA availability...${NC}"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  requirements.txt not found, skipping dependency installation${NC}"
fi

# Create output directory
mkdir -p outputs/$EXPERIMENT_NAME

# Step 1: Download and analyze model (optional - can be skipped if model already cached)
echo -e "${GREEN}📥 Step 1: Downloading and analyzing model...${NC}"
python3 scripts/download_model.py \
    --model $MODEL_NAME \
    --config configs/model_configs/phi2_diffusion.yaml \
    --cache-dir ./model_cache

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Model download/analysis failed, but continuing...${NC}"
fi

# Step 2: Run dry run to check setup
echo -e "${GREEN}🧪 Step 2: Running setup validation (dry run)...${NC}"
python3 scripts/train_diffusion.py \
    --model_config configs/model_configs/phi2_diffusion.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml \
    --dry_run

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Dry run failed! Please check your configuration.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Dry run successful!${NC}"
echo ""

# Ask user if they want to proceed with actual training
read -p "$(echo -e ${YELLOW}Do you want to proceed with actual training? This may take several hours. [y/N]: ${NC})" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Training cancelled by user.${NC}"
    exit 0
fi

# Step 3: Run conversion training
echo -e "${GREEN}🏃 Step 3: Starting conversion training...${NC}"
echo -e "${YELLOW}This may take several hours depending on your hardware.${NC}"
echo -e "${YELLOW}Monitor progress with: tail -f outputs/$EXPERIMENT_NAME/training_logs.json${NC}"
echo ""

python3 scripts/train_diffusion.py \
    --model_config configs/model_configs/phi2_diffusion.yaml \
    --training_config configs/training_configs/stage2_diffusion.yaml \
    --hardware_config configs/hardware_configs/single_a6000.yaml

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Training completed successfully!${NC}"

# Step 4: Evaluate converted model (if evaluation script exists)
if [ -f "scripts/evaluate.py" ]; then
    echo -e "${GREEN}📊 Step 4: Evaluating converted model...${NC}"
    python3 scripts/evaluate.py \
        --checkpoint outputs/$EXPERIMENT_NAME/best_checkpoint \
        --output_dir outputs/$EXPERIMENT_NAME/evaluation
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Evaluation completed!${NC}"
    else
        echo -e "${YELLOW}⚠️  Evaluation failed, but training was successful.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Evaluation script not found, skipping evaluation.${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Conversion pipeline completed!${NC}"
echo -e "${BLUE}Results saved to: outputs/$EXPERIMENT_NAME/${NC}"
echo -e "${BLUE}Final model: outputs/$EXPERIMENT_NAME/final_model/${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "${YELLOW}1. Check the training logs: outputs/$EXPERIMENT_NAME/training_logs.json${NC}"
echo -e "${YELLOW}2. Test the converted model with your own prompts${NC}"
echo -e "${YELLOW}3. Fine-tune further if needed${NC}"
echo ""
echo -e "${GREEN}Happy diffusing! 🌊${NC}"