"""
Inspect GRPO Data
Shows the prompts prepared for Reinforcement Learning
"""

from train_grpo import GRPOTrainingPipeline

def inspect_data():
    print("🔍 Preparing GRPO Dataset...")
    pipeline = GRPOTrainingPipeline()
    dataset = pipeline.prepare_dataset()
    
    print(f"\n✅ Dataset Loaded. Total samples: {len(dataset)}")
    print("=" * 80)
    print("SAMPLE PROMPTS (Input to GRPO)")
    print("=" * 80)
    
    for i in range(min(3, len(dataset))):
        print(f"\n--- Sample {i+1} ---")
        print(dataset[i]['prompt'])
        print("-" * 40)

if __name__ == "__main__":
    inspect_data()

