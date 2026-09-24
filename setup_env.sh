#!/bin/bash
set -e

# Initialize conda
source ~/miniconda3/bin/activate

# Accept TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create normal work environment (Python 3.10)
echo "Setting up 'acc' environment (Python 3.10)..."
conda create -n acc python=3.10 -y || true
conda activate acc
pip install -r requirements.txt huggingface_hub
conda deactivate

# Create benchmarks environment (Python 3.11)
echo "Setting up 'acc_bench' environment (Python 3.11)..."
conda create -n acc_bench python=3.11 -y || true
conda activate acc_bench
pip install -r requirements.txt huggingface_hub

# Note: We keep acc_bench activated for the rest of the script so huggingface-cli is available

# Setup data directories
mkdir -p DATA/Benchmarks/POPE DATA/Benchmarks/MathVista DATA/Benchmarks/ALFWorld DATA/Benchmarks/VQAv2 DATA/models

# Download Benchmarks
echo "Downloading POPE..."
if [ ! -d "DATA/Benchmarks/POPE/.git" ] && [ ! -f "DATA/Benchmarks/POPE/README.md" ]; then
    git clone https://github.com/AoiDragon/POPE.git DATA/Benchmarks/POPE_tmp
    cp -r DATA/Benchmarks/POPE_tmp/* DATA/Benchmarks/POPE/
    rm -rf DATA/Benchmarks/POPE_tmp
fi

echo "Downloading MathVista..."
hf download AI4Math/MathVista --repo-type dataset --local-dir DATA/Benchmarks/MathVista

echo "Downloading ALFWorld..."
if [ ! -d "DATA/Benchmarks/ALFWorld/.git" ] && [ ! -f "DATA/Benchmarks/ALFWorld/README.md" ]; then
    git clone https://github.com/alfworld/alfworld.git DATA/Benchmarks/ALFWorld_tmp
    cp -r DATA/Benchmarks/ALFWorld_tmp/* DATA/Benchmarks/ALFWorld/
    rm -rf DATA/Benchmarks/ALFWorld_tmp
fi

echo "Downloading VQAv2 Annotations and Questions..."
if [ ! -f "DATA/Benchmarks/VQAv2/v2_Questions_Train_mscoco.zip" ]; then
    wget -q https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P DATA/Benchmarks/VQAv2/
    wget -q https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P DATA/Benchmarks/VQAv2/
    wget -q https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P DATA/Benchmarks/VQAv2/
    wget -q https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P DATA/Benchmarks/VQAv2/
    echo "VQAv2 zipped annotations downloaded (Note: MSCOCO images must be downloaded manually if required)."
fi

# Download models using provided HF token (Provide your own token here)
export HF_TOKEN="your_hf_token_here"
# hf auth login --token $HF_TOKEN

echo "Downloading models..."
hf download llava-hf/llava-v1.6-mistral-7b-hf --local-dir DATA/models/llava-v1.6-mistral-7b-hf
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir DATA/models/Qwen2.5-VL-3B-Instruct
hf download microsoft/Phi-4-multimodal-instruct --local-dir DATA/models/Phi-4-multimodal-instruct
hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir DATA/models/Meta-Llama-3-8B-Instruct

echo "Setup completed successfully."
