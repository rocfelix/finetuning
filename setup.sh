#!/bin/bash

# Setup script for LLM Fine-tuning Project on Mac M1/M2/M3
# This script will set up your environment and install all dependencies using uv

set -e  # Exit on error

echo "🚀 Setting up LLM Fine-tuning Project for Apple Silicon..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📥 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install uv. Please install it manually:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
fi

# Check if we're on Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "⚠️  Warning: This setup is optimized for Apple Silicon (M1/M2/M3)"
    echo "   Detected architecture: $ARCH"
else
    echo "✅ Running on Apple Silicon ($ARCH)"
fi

# Install dependencies with uv
echo ""
echo "📦 Installing dependencies with uv..."
uv sync

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p outputs
mkdir -p model_cache

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Run commands with uv: uv run python main.py prepare-data"
echo "   2. Or activate environment: source .venv/bin/activate"
echo "   3. Review and edit config.yaml for your needs"
echo "   4. Test data preparation: uv run python main.py prepare-data"
echo "   5. Start training: uv run python main.py train"
echo ""
echo "📖 For detailed instructions, see README.md"
echo ""
echo "💡 Tip: For M1 Macs, start with smaller models like:"
echo "   - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B params)"
echo "   - microsoft/phi-2 (2.7B params)"
echo ""
echo "⚡ uv advantages:"
echo "   - 10-100x faster than pip"
echo "   - Automatic environment management"
echo "   - Better dependency resolution"
echo ""

