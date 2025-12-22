
import argparse
import sys
import os
import yaml
from inference import LLMInference

def compare_models(config_path, adapter_path, instruction, input_text=""):
    """
    Compare responses between the base model and the fine-tuned model.
    """
    print(f"📊 Comparing Base Model vs Fine-Tuned Model")
    print(f"   Config: {config_path}")
    print(f"   Adapter: {adapter_path}")
    print(f"   Instruction: {instruction}")
    if input_text:
        print(f"   Input: {input_text}")
    print("-" * 60)

    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    base_model_name = config['model']['name']

    # 1. Generate with Base Model
    print("\n🔵 Loading Base Model...")
    inference = LLMInference(base_model_path=base_model_name, adapter_path=None, config_path=config_path)
    
    print("\n⏳ Generating Base Model Response...")
    response_base = inference.generate(instruction, input_text, max_new_tokens=256)

    # 2. Generate with Fine-Tuned Model
    print("\n🟣 Loading Fine-Tuned Model...")
    # We can reuse the inference object but need to reload the model with the adapter
    if os.path.exists(adapter_path):
        inference.load_model(base_model_name, adapter_path=adapter_path)
        print("\n⏳ Generating Fine-Tuned Response...")
        response_ft = inference.generate(instruction, input_text, max_new_tokens=256)
    else:
        print(f"\n⚠️  Adapter path '{adapter_path}' not found.")
        print("   Cannot generate fine-tuned response. Please train the model first.")
        response_ft = "N/A (Model not trained)"

    # 3. Display Comparison
    print("\n" + "=" * 80)
    print("📝 COMPARISON RESULTS")
    print("=" * 80)
    print(f"INSTRUCTION: {instruction}")
    if input_text:
        print(f"INPUT:       {input_text}")
    print("-" * 80)
    
    print("🔵 BASE MODEL RESPONSE:")
    print(response_base)
    print("-" * 80)
    
    print("🟣 FINE-TUNED MODEL RESPONSE:")
    print(response_ft)
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Compare Base vs Fine-Tuned Model Output")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--adapter_path", type=str, default="./outputs/final_model", help="Path to trained adapter")
    parser.add_argument("--instruction", type=str, required=True, help="Instruction to test")
    parser.add_argument("--input", type=str, default="", help="Optional input context")
    
    args = parser.parse_args()
    
    compare_models(args.config, args.adapter_path, args.instruction, args.input)

if __name__ == "__main__":
    main()
