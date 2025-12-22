"""
LLM Supervised Fine-Tuning (SFT) Project - Main Entry Point

This is the main entry point for the SFT project. You can run different components:
- Data preparation
- Model training
- Inference and testing

For detailed usage, see README.md
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="LLM Fine-tuning Project - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data
  python main.py prepare-data
  
  # Train model
  python main.py train
  
  # Run inference (chat mode)
  python main.py inference
  
  # Run inference (single query)
  python main.py inference --instruction "Explain AI" --no-chat

For more details, see README.md
        """
    )

    parser.add_argument(
        'command',
        choices=['prepare-data', 'train', 'inference', 'compare', 'train-grpo'],
        help='Command to run'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    # Inference specific arguments
    parser.add_argument(
        '--base_model',
        type=str,
        help='Base model for inference'
    )

    parser.add_argument(
        '--adapter_path',
        type=str,
        default='./outputs/final_model',
        help='Path to LoRA adapter'
    )

    parser.add_argument(
        '--instruction',
        type=str,
        help='Single instruction for inference or comparison'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default="",
        help='Input context for comparison'
    )

    parser.add_argument(
        '--no-chat',
        action='store_true',
        help='Disable chat mode in inference'
    )

    args = parser.parse_args()

    if args.command == 'prepare-data':
        print("🔄 Preparing data...")
        from data_preparation import DataPreparation

        data_prep = DataPreparation(args.config)
        train_dataset, eval_dataset = data_prep.prepare_dataset(format_style="alpaca")
        
        # Save preview of both datasets
        data_prep.save_dataset_preview(train_dataset, eval_dataset)
        
        # Generate visualizations
        data_prep.visualize_dataset(train_dataset, eval_dataset)

        print("✅ Data preparation complete!")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Eval samples: {len(eval_dataset)}")
        print(f"   Preview saved to: dataset_preview.txt")
        print(f"   Visualizations saved to: outputs/plots/")

    elif args.command == 'train':
        print("🚀 Starting training...")
        from train import LLMTrainer

        trainer = LLMTrainer(args.config)
        trainer.train()

        print("✅ Training complete!")

    elif args.command == 'train-grpo':
        print("🧠 Starting GRPO (RL) training...")
        from train_grpo import GRPOTrainingPipeline
        
        pipeline = GRPOTrainingPipeline(args.config)
        pipeline.train()
        
        print("✅ GRPO Training complete!")

    elif args.command == 'inference':
        print("💬 Starting inference...")
        from inference import LLMInference

        # Get base model from args or config
        if args.base_model:
            base_model = args.base_model
        else:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            base_model = config['model']['name']

        inference = LLMInference(
            base_model_path=base_model,
            adapter_path=args.adapter_path,
            config_path=args.config
        )

        if args.instruction and args.no_chat:
            response = inference.generate(instruction=args.instruction)
            print(f"\n📝 Instruction: {args.instruction}")
            print(f"🤖 Response: {response}\n")
        else:
            inference.chat()
            
    elif args.command == 'compare':
        if not args.instruction:
            print("❌ Error: --instruction is required for comparison mode.")
            return

        from compare import compare_models
        compare_models(args.config, args.adapter_path, args.instruction, args.input)


if __name__ == '__main__':
    main()
