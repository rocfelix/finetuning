"""
Example usage scripts for the LLM Fine-tuning Project
Run these examples to get started quickly
"""

import yaml
from data_preparation import DataPreparation
from inference import LLMInference


def example_1_test_data_preparation():
    """Example 1: Test data preparation with default dataset."""
    print("=" * 60)
    print("Example 1: Testing Data Preparation")
    print("=" * 60)

    data_prep = DataPreparation()

    # Load and prepare dataset
    train_dataset, eval_dataset = data_prep.prepare_dataset(format_style="alpaca")

    print(f"\n✅ Dataset prepared successfully!")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")

    # Show a sample
    print(f"\n📝 Sample training example:")
    print("-" * 60)
    print(train_dataset[0]['text'][:500] + "...")
    print("-" * 60)

    # Save samples to file
    data_prep.save_sample_prompts(train_dataset, n_samples=3)
    print(f"\n💾 Sample prompts saved to 'sample_prompts.txt'")


def example_2_custom_dataset():
    """Example 2: Create and use a custom dataset."""
    print("\n" + "=" * 60)
    print("Example 2: Using Custom Dataset")
    print("=" * 60)

    # Create a small custom dataset
    import json

    custom_data = [
        {
            "instruction": "What is machine learning?",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "Explain the following concept in simple terms",
            "input": "Neural networks",
            "output": "Neural networks are computing systems inspired by the human brain. They consist of interconnected nodes (like neurons) that process information in layers, learning patterns from data to make predictions or decisions."
        },
        {
            "instruction": "Write a haiku about coding",
            "input": "",
            "output": "Lines of code flow free\nBugs hide in silent shadows\nDebugger finds truth"
        }
    ]

    # Save to file
    custom_file = "custom_dataset.json"
    with open(custom_file, 'w') as f:
        json.dump(custom_data, f, indent=2)

    print(f"✅ Created custom dataset: {custom_file}")

    # Load and prepare custom dataset
    data_prep = DataPreparation()
    dataset = data_prep.load_custom_dataset(custom_file)

    # Format it
    formatted_dataset = dataset.map(
        data_prep.format_alpaca_style,
        batched=True,
        remove_columns=dataset.column_names
    )

    print(f"\n📝 Formatted custom dataset sample:")
    print("-" * 60)
    print(formatted_dataset[0]['text'])
    print("-" * 60)


def example_3_config_modification():
    """Example 3: Show how to modify configuration for different scenarios."""
    print("\n" + "=" * 60)
    print("Example 3: Configuration Examples")
    print("=" * 60)

    # Load current config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\n🔧 Configuration Scenarios:\n")

    # Scenario 1: Quick test with tiny model
    print("1. QUICK TEST (Recommended for first run):")
    print("   model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("   num_train_epochs: 1")
    print("   per_device_train_batch_size: 2")
    print("   max_samples: 100")
    print("   ⏱️  Expected time: ~15-30 minutes")

    # Scenario 2: Balanced training
    print("\n2. BALANCED TRAINING (Good for M1 with 16GB RAM):")
    print("   model: microsoft/phi-2 or mistralai/Mistral-7B-v0.1")
    print("   num_train_epochs: 3")
    print("   per_device_train_batch_size: 2")
    print("   gradient_accumulation_steps: 4")
    print("   ⏱️  Expected time: ~4-8 hours")

    # Scenario 3: Full fine-tuning
    print("\n3. FULL FINE-TUNING (M1 Max/Ultra with 32GB+ RAM):")
    print("   model: meta-llama/Llama-2-7b-hf")
    print("   num_train_epochs: 3")
    print("   per_device_train_batch_size: 4")
    print("   gradient_accumulation_steps: 4")
    print("   ⏱️  Expected time: ~8-12 hours")

    print("\n💡 Tips:")
    print("   - Start with Scenario 1 to test your setup")
    print("   - Monitor memory usage with Activity Monitor")
    print("   - Reduce batch_size if you run out of memory")
    print("   - Enable wandb logging to track experiments")


def example_4_inference_testing():
    """Example 4: Show how to test inference without training."""
    print("\n" + "=" * 60)
    print("Example 4: Testing Inference (Simulation)")
    print("=" * 60)

    print("\n📝 To test a trained model, use these commands:\n")

    print("1. Interactive chat mode:")
    print("   python inference.py --base_model [MODEL_NAME] --adapter_path ./outputs/final_model")

    print("\n2. Single query:")
    print("   python inference.py \\")
    print("     --base_model [MODEL_NAME] \\")
    print("     --adapter_path ./outputs/final_model \\")
    print("     --instruction 'Explain quantum computing'")

    print("\n3. Using main.py:")
    print("   python main.py inference")

    print("\n💡 Note: You need to train a model first or use a pre-trained adapter")


def example_5_training_walkthrough():
    """Example 5: Complete training walkthrough."""
    print("\n" + "=" * 60)
    print("Example 5: Complete Training Walkthrough")
    print("=" * 60)

    print("\n📋 Step-by-step guide:\n")

    print("Step 1: Setup environment")
    print("   chmod +x setup.sh")
    print("   ./setup.sh  # Installs uv and dependencies")

    print("\nStep 2: Prepare data")
    print("   uv run python main.py prepare-data")
    print("   # Review sample_prompts.txt")

    print("\nStep 3: Edit config.yaml")
    print("   # Set model to TinyLlama for first test")
    print("   # Adjust batch_size based on your RAM")

    print("\nStep 4: Start training")
    print("   uv run python main.py train")
    print("   # Monitor progress in terminal")

    print("\nStep 5: Check training progress")
    print("   uv run tensorboard --logdir outputs/logs")
    print("   # Open http://localhost:6006 in browser")

    print("\nStep 6: Test your model")
    print("   uv run python main.py inference")

    print("\n⏱️  Timeline:")
    print("   - Setup: 2-3 minutes (uv is fast!)")
    print("   - Data prep: 2-5 minutes")
    print("   - Training (TinyLlama): 15-30 minutes")
    print("   - Training (Mistral-7B): 4-8 hours")

    print("\n💡 Note: All commands use 'uv run' - no environment activation needed!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LLM Fine-tuning Project - Examples & Tutorials")
    print("=" * 60)

    examples = {
        '1': ('Test Data Preparation', example_1_test_data_preparation),
        '2': ('Custom Dataset', example_2_custom_dataset),
        '3': ('Configuration Guide', example_3_config_modification),
        '4': ('Inference Testing', example_4_inference_testing),
        '5': ('Training Walkthrough', example_5_training_walkthrough),
    }

    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"   {key}. {name}")
    print("   0. Run all examples")
    print("   q. Quit")

    while True:
        choice = input("\nSelect an example (0-5, q to quit): ").strip()

        if choice == 'q':
            print("Goodbye!")
            break
        elif choice == '0':
            for key in sorted(examples.keys()):
                examples[key][1]()
            break
        elif choice in examples:
            examples[choice][1]()
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

