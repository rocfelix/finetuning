# LLM Supervised Fine-Tuning (SFT) Project

A complete implementation for fine-tuning Large Language Models using Supervised Fine-Tuning (SFT) with LoRA/QLoRA, optimized for **Apple Silicon (M1/M2/M3)**.

## 🚀 Features

- **Parameter-Efficient Fine-Tuning** with LoRA (Low-Rank Adaptation)
- **Quantization Support** (4-bit/8-bit) for memory efficiency (Linux/CUDA only)
- **Apple Silicon Optimized** - Leverages MPS backend for M1/M2/M3 chips (uses float16 instead of quantization)
- **Fast Package Management** - Uses `uv` for 10-100x faster installs
- **Flexible Data Loading** - Support for HuggingFace datasets and custom data
- **Multiple Prompt Formats** - Alpaca and Chat styles
- **Interactive Inference** - Chat interface for testing models
- **Monitoring** - TensorBoard and Weights & Biases integration

## 📋 Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- 16GB+ RAM recommended (8GB minimum for smaller models)
- `uv` package manager

## 🔧 Installation

1. **Clone or navigate to this project directory**

2. **Run the setup script** (installs `uv` and all dependencies)
```bash
./setup.sh
```

Alternatively, if you have `uv` installed:
```bash
uv sync
```

## 📁 Project Structure

```
.
├── config.yaml              # Main configuration file
├── pyproject.toml           # Project configuration & dependencies
├── data_preparation.py      # Data loading and preprocessing
├── train.py                 # Training script
├── inference.py            # Inference and chat interface
├── README.md               # This file
├── outputs/                # Training outputs (created automatically)
│   ├── final_model/        # Saved model and adapter
│   └── logs/               # TensorBoard logs
└── model_cache/            # Downloaded models cache
```

## 🎯 Quick Start

### 1. Configure Your Training

Edit `config.yaml` to customize:
- Model selection (default: Llama-2-7b)
- LoRA parameters
- Training hyperparameters
- Dataset selection

For M1 Macs, consider starting with smaller models:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B parameters)
- `microsoft/phi-2` (2.7B parameters)
- `mistralai/Mistral-7B-v0.1` (7B parameters)

### 2. Prepare Your Data

**Option A: Use a HuggingFace Dataset** (default: Alpaca cleaned)
```python
# Already configured in config.yaml
dataset:
  name: "yahma/alpaca-cleaned"
```

**Option B: Use Custom Data**

Create a JSON file with this format:
```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  },
  {
    "instruction": "Translate the following to Spanish",
    "input": "Hello, how are you?",
    "output": "Hola, ¿cómo estás?"
  }
]
```

Test data preparation:
```bash
uv run python main.py prepare-data
```

This will:
1. Download and format the dataset
2. Create `sample_prompts.txt` for inspection
3. **Generate visualizations** in `outputs/plots/` (token length distribution)
   - *Check this to ensure your `max_seq_length` covers most data!*

### 3. Train Your Model

**Standard SFT (Supervised Fine-Tuning):**
```bash
uv run python main.py train
```

**Reinforcement Learning (GRPO):**
For reasoning tasks or when you have a reward function (e.g., preference optimization without a critic):
```bash
uv run python main.py train-grpo
```

**Understanding GRPO:**
Group Relative Policy Optimization (GRPO) improves the model by generating multiple outputs for a prompt and reinforcing the ones that score higher on a reward function. It avoids the need for a separate "critic" model (like in PPO), making it more memory-efficient.

**Configuration (`config.yaml`):**
Configure GRPO in the `grpo:` section:
- `reward_function`: Choose the logic to score responses.
  - `tool_use`: (Default) Rewards correct JSON function calls (great for agentic tasks).
  - `format`: Rewards using specific XML tags (e.g., `<think>`...`</think>`).
  - `length`: Simple proxy rewarding longer, more detailed responses.
  - `accuracy`: Basic check (penalizes "I don't know").
- `num_generations`: How many variations to generate per prompt (higher = better stability but more memory).
- `beta`: Controls how much the model can deviate from the original behavior (KL penalty).

*Note: GRPO uses the same dataset source as SFT but processes it differently to optimize for the chosen reward.*

**Inspect Data:**
To see exactly what prompts are being fed into the GRPO trainer:
```bash
uv run python inspect_grpo_data.py
```

**Training Tips for M1:**
- Start with `batch_size: 1` or `2` and adjust based on memory
- Use `gradient_accumulation_steps: 4-8` to simulate larger batches
- Enable `gradient_checkpointing: true` to save memory
- Monitor memory usage with Activity Monitor
- **Note:** 4-bit quantization (`load_in_4bit`) is **disabled** on macOS. See `MACOS_NOTES.md` for details.

Expected training time:

### 4. Test Your Model

**Interactive Chat:**
```bash
uv run python main.py inference
```

**Single Query:**
```bash
uv run python main.py inference --instruction "Explain quantum computing in simple terms" --no-chat
```

### 5. ⚖️ Compare Before & After

You can instantly see the difference between the base model and your fine-tuned model:

```bash
uv run python main.py compare --instruction "Explain AI"
```

Or with input context:
```bash
uv run python main.py compare \
  --instruction "Summarize this text" \
  --input "Artificial intelligence is intelligence demonstrated by machines..."
```

This will load both models and display their responses side-by-side.

## ⚙️ Configuration Templates

### Important: bitsandbytes Not Available on macOS

The `bitsandbytes` library (used for 4-bit/8-bit quantization) **does not support macOS**. This is a known limitation.

**What This Means:**
- ❌ **Cannot use 4-bit/8-bit quantization** on Mac
- ✅ **Can still train models** using float16 precision
- ✅ **LoRA fine-tuning still works perfectly**
- ⚠️ **Higher memory usage** compared to quantized training

### Automatic Handling

The project automatically detects macOS and:
1. Skips bitsandbytes installation
2. Uses `dtype=float16` instead of 4-bit quantization
3. Logs a warning when quantization is requested but not available

### Configuration for macOS

In `config.yaml`, `load_in_4bit` is set to `false`:

```yaml
quantization:
  load_in_4bit: false  # Must be false on macOS
```

## ⚙️ Configuration Templates

Here are optimized configurations for different scenarios. Copy these into your `config.yaml`.

### 1. QUICK TEST (15-30 mins)
Fastest way to test setup.
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset:
  max_samples: 100
training:
  num_train_epochs: 1
```

### 2. BALANCED - M1 Mac (8-16GB RAM)
Good balance for general M1.
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_seq_length: 512
```

### 3. QUALITY - M1 Pro/Max (16-32GB RAM)
Higher quality with larger model.
```yaml
model:
  name: "mistralai/Mistral-7B-v0.1"
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 1024
```

### 4. PRODUCTION - M1 Ultra/Max (32GB+ RAM)
Best quality.
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
training:
  num_train_epochs: 5
  per_device_train_batch_size: 4
  max_seq_length: 2048
```

## 📊 Monitoring Training

**TensorBoard** (default):
```bash
uv run tensorboard --logdir outputs/logs
```

**Weights & Biases** (optional):
1. Set `use_wandb: true` in `config.yaml`
2. Login: `uv run wandb login`
3. View at wandb.ai

## 🛠️ Development Workflow

This project uses `uv` for dependency management. Common commands:

- **Add a package:** `uv add package_name`
- **Add dev package:** `uv add --dev pytest`
- **Update all:** `uv sync --upgrade`
- **Run command:** `uv run python script.py`
- **Export requirements:** `uv pip compile pyproject.toml -o requirements.txt`

## 🐛 Troubleshooting

### Memory Issues
- **Error: "Out of memory"**
  - Reduce `per_device_train_batch_size` to 1
  - Increase `gradient_accumulation_steps`
  - Reduce `max_seq_length` to 256 or 384
  - Close other applications

### Installation Issues
- **BitsAndBytes:** Not supported on macOS. Float16 is used automatically.
- **Transformers Version:** `uv sync --refresh`

### Model Loading Issues
- **Token Auth:** `uv run huggingface-cli login`
- **Cache:** `rm -rf model_cache/`

## 📚 Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [uv Documentation](https://github.com/astral-sh/uv)

## 📄 License
This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments
Built with [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [uv](https://github.com/astral-sh/uv).

---
**Happy Fine-tuning! 🎉**