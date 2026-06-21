#!/bin/bash
# FRIDAY setup, run once
set -e
cd "$(dirname "$0")/.."

echo "→ Checking Python 3.11+..."
command -v python3.11 >/dev/null || { echo "Install Python 3.11: brew install python@3.11"; exit 1; }

echo "→ Creating venv..."
python3.11 -m venv daemon/.venv
source daemon/.venv/bin/activate

echo "→ Upgrading pip..."
pip install -q --upgrade pip wheel setuptools

echo "→ Installing requirements (this takes 5-10 min)..."
pip install -r daemon/requirements.txt

echo "→ Building OCR helper..."
xcrun swiftc -O daemon/bin/ocr.swift -o daemon/bin/ocr

echo "→ Pre-downloading models (this takes 10-20 min, ~12GB)..."
python3 -c "
from mlx_lm import load
print('Downloading Qwen3-14B (brain, ~8GB)...')
load('mlx-community/Qwen3-14B-4bit')
print('Downloading Qwen3-VL-4B (vision fallback, ~3GB)...')
from mlx_vlm import load as vl_load
vl_load('mlx-community/Qwen3-VL-4B-Instruct-4bit')
print('Downloading embedder...')
from sentence_transformers import SentenceTransformer
SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
print('Downloading Whisper small.en and tiny.en...')
from pywhispercpp.model import Model
Model('small.en')
Model('tiny.en')
print('Downloading Kokoro...')
from kokoro import KPipeline
KPipeline(lang_code='a')
print('All models ready.')
"

echo ""
echo "✓ Setup complete."
echo "  Run daemon:  ./scripts/run.sh"
echo "  Open Xcode and build the FRIDAY.app target."
