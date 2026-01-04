"""
Data Preparation Script for SFT
Handles loading and preprocessing datasets for instruction tuning
"""

import json
import yaml
import os
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer


class DataPreparation:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data preparation with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.training_config = self.config['training']
        self.model_config = self.config.get('model', {})
    
    def load_dataset_from_hub(self) -> Dataset:
        """Load dataset from Hugging Face Hub."""
        print(f"Loading dataset: {self.dataset_config['name']}")
        
        load_args = [self.dataset_config['name']]
        if self.dataset_config.get('config_name'):
            load_args.append(self.dataset_config['config_name'])
            
        dataset = load_dataset(
            *load_args,
            split=self.dataset_config['split']
        )
        
        # Limit samples if specified (useful for testing)
        if self.dataset_config['max_samples']:
            dataset = dataset.select(range(self.dataset_config['max_samples']))
        
        return dataset
    
    def load_custom_dataset(self, filepath: str) -> Dataset:
        """
        Load custom dataset from JSON or JSONL file.
        Expected format: [{"instruction": "...", "input": "...", "output": "..."}]
        """
        print(f"Loading custom dataset from: {filepath}")
        
        try:
            if filepath.endswith('.jsonl'):
                data = []
                with open(filepath, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
            else:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            
            if not data:
                raise ValueError(f"Dataset file {filepath} is empty")
                
            return Dataset.from_list(data)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from {filepath}: {e}")
    
    def format_alpaca_style(self, examples: Dict) -> Dict:
        """
        Format examples in Alpaca instruction style.
        Creates prompts from instruction, input, and output fields using config templates.
        """
        prompts = []
        
        # Get templates from config or use hardcoded fallbacks
        templates = self.config.get('templates', {})
        alpaca_template = templates.get('alpaca', {})
        
        # Define default templates if missing from config
        default_with_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
        default_no_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        template_with_input = alpaca_template.get('with_input', default_with_input)
        template_no_input = alpaca_template.get('no_input', default_no_input)

        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples.get('input', [''] * len(examples['instruction']))[i]
            output = examples['output'][i]
            
            # Build prompt
            if input_text:
                prompt = template_with_input.format(
                    instruction=instruction,
                    input=input_text
                )
            else:
                prompt = template_no_input.format(
                    instruction=instruction
                )
            
            # Append output
            prompt += f"{output}"
            
            prompts.append(prompt)
        
        return {"text": prompts}
    
    def format_chat_style(self, examples: Dict) -> Dict:
        """
        Format examples in chat/conversation style.
        Uses simpler user/assistant format.
        """
        prompts = []
        
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples.get('input', [''] * len(examples['instruction']))[i]
            output = examples['output'][i]
            
            # Build prompt
            user_message = f"{instruction}\n{input_text}" if input_text else instruction
            prompt = f"<|user|>\n{user_message}\n<|assistant|>\n{output}"
            
            prompts.append(prompt)
        
        return {"text": prompts}
    
    def format_glaive_style(self, examples: Dict) -> Dict:
        """
        Format examples from Glaive function calling dataset.
        Expected columns: 'system', 'chat'
        """
        prompts = []
        
        for i in range(len(examples['chat'])):
            system_prompt = examples.get('system', [''])[i]
            chat_history = examples['chat'][i]
            
            # Glaive v2 usually has chat as a string with "USER: ... ASSISTANT: ..."
            # or it might be a list of dicts.
            # Let's handle string format which is common in raw datasets
            
            prompt = ""
            if system_prompt:
                prompt += f"System: {system_prompt}\n\n"
            
            prompt += chat_history
            
            prompts.append(prompt)
        
        return {"text": prompts}

    def prepare_dataset(
        self, 
        dataset: Optional[Dataset] = None,
        format_style: str = "alpaca"
    ) -> tuple:
        """
        Prepare dataset for training.
        
        Args:
            dataset: Dataset to prepare. If None, loads from config.
            format_style: Either "alpaca" or "chat"
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if dataset is None:
            dataset = self.load_dataset_from_hub()
        
        print(f"Dataset size: {len(dataset)}")
        
        # Format the dataset
        if "glaive" in self.dataset_config['name'].lower():
             formatted_dataset = dataset.map(
                self.format_glaive_style,
                batched=True,
                remove_columns=dataset.column_names
            )
        elif format_style == "alpaca":
            formatted_dataset = dataset.map(
                self.format_alpaca_style,
                batched=True,
                remove_columns=dataset.column_names
            )
        elif format_style == "chat":
            formatted_dataset = dataset.map(
                self.format_chat_style,
                batched=True,
                remove_columns=dataset.column_names
            )
        else:
            raise ValueError(f"Unknown format_style: {format_style}")
        
        # Split into train and eval
        split_dataset = formatted_dataset.train_test_split(
            test_size=self.dataset_config['test_size'],
            seed=self.dataset_config['seed']
        )
        
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def save_dataset_preview(self, train_dataset: Dataset, eval_dataset: Dataset, output_file: str = "dataset_preview.txt", n_samples: int = 3):
        """Save preview of train and eval datasets for inspection."""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"DATASET PREVIEW (Format: {self.config.get('templates', {}).get('active_style', 'alpaca')})\n")
            f.write("="*80 + "\n\n")

            f.write("="*40 + "\n")
            f.write("TRAIN DATASET (First {} samples)\n".format(n_samples))
            f.write("="*40 + "\n\n")
            for i in range(min(n_samples, len(train_dataset))):
                f.write(f"--- Train Sample {i+1} ---\n")
                f.write(train_dataset[i]['text'])
                f.write("\n" + "-"*40 + "\n\n")
            
            f.write("\n" + "="*40 + "\n")
            f.write("EVAL DATASET (First {} samples)\n".format(n_samples))
            f.write("="*40 + "\n\n")
            for i in range(min(n_samples, len(eval_dataset))):
                f.write(f"--- Eval Sample {i+1} ---\n")
                f.write(eval_dataset[i]['text'])
                f.write("\n" + "-"*40 + "\n\n")
                
        print(f"Dataset preview saved to {output_file}")

    def visualize_dataset(self, train_dataset: Dataset, eval_dataset: Dataset, output_dir: str = "outputs/plots"):
        """
        Visualize the token length distribution of the datasets.
        """
        print("📊 Generating dataset visualizations...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer
        model_name = self.model_config.get('name', 'meta-llama/Llama-2-7b-hf')
        cache_dir = self.model_config.get('cache_dir', './model_cache')
        auth_token = self.model_config.get('use_auth_token')
        
        print(f"   Loading tokenizer for {model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                token=auth_token,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"   ⚠️ Could not load tokenizer: {e}")
            print("   Using simple whitespace splitting for length estimation.")
            tokenizer = None

        def get_lengths(dataset):
            lengths = []
            for item in dataset:
                if tokenizer:
                    lengths.append(len(tokenizer.encode(item['text'])))
                else:
                    lengths.append(len(item['text'].split()))
            return lengths

        print("   Calculating token lengths...")
        train_lengths = get_lengths(train_dataset)
        eval_lengths = get_lengths(eval_dataset)
        
        # Create DataFrame for plotting
        data = []
        for l in train_lengths:
            data.append({'Length': l, 'Split': 'Train'})
        for l in eval_lengths:
            data.append({'Length': l, 'Split': 'Eval'})
        
        df = pd.DataFrame(data)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Length', hue='Split', bins=50, kde=True, element="step")
        
        max_seq_len = self.training_config.get('max_seq_length', 512)
        plt.axvline(x=max_seq_len, color='r', linestyle='--', label=f'Max Seq Len ({max_seq_len})')
        
        plt.title(f'Token Length Distribution ({model_name})')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        plt.legend()
        
        output_path = os.path.join(output_dir, "token_distribution.png")
        plt.savefig(output_path)
        print(f"   ✅ Saved plot to {output_path}")
        
        # Print stats
        print("\n   📈 Statistics:")
        print(f"   Train: Mean={pd.Series(train_lengths).mean():.1f}, Max={max(train_lengths)}")
        print(f"   Eval:  Mean={pd.Series(eval_lengths).mean():.1f}, Max={max(eval_lengths)}")
        print(f"   Max Seq Length Config: {max_seq_len}")
        
        pct_truncated = sum(1 for l in train_lengths if l > max_seq_len) / len(train_lengths) * 100
        print(f"   ⚠️ {pct_truncated:.1f}% of train samples will be truncated")


if __name__ == "__main__":
    # Example usage
    data_prep = DataPreparation()
    
    # Load and prepare dataset
    train_dataset, eval_dataset = data_prep.prepare_dataset(format_style="alpaca")
    
    # Save sample prompts
    data_prep.save_dataset_preview(train_dataset, eval_dataset)
    
    print("Data preparation complete!")

