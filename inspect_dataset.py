
from datasets import load_dataset

try:
    dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
    print("Dataset loaded successfully")
    print("Columns:", next(iter(dataset)).keys())
    print("Sample:", next(iter(dataset)))
except Exception as e:
    print(f"Error loading dataset: {e}")
