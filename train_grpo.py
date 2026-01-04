"""
GRPO Training Script
Implements Group Relative Policy Optimization for RLHF/RLAIF
"""

import os
import re
import json
import yaml
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from data_preparation import DataPreparation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRPOTrainingPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model_config = self.config['model']
        self.grpo_config = self.config['grpo']
        self.lora_config = self.config['lora']
        self.training_config = self.config['training']
        self.dataset_config = self.config['dataset']
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        model_name = self.model_config['name']
        logger.info(f"Loading model: {model_name}")
        
        # Load kwargs optimized for Mac/MPS
        load_kwargs = {
            'cache_dir': self.model_config['cache_dir'],
            'trust_remote_code': True,
            'token': self.model_config.get('use_auth_token'),
            'torch_dtype': torch.bfloat16,  # Use bfloat16 for better stability
            'low_cpu_mem_usage': False # Avoid meta tensors on MPS
        }
        
        if self.device.type == 'mps':
            # Avoid device_map="auto" on MPS
            pass
        else:
            load_kwargs['device_map'] = 'auto'

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if self.device.type == 'mps':
            self.model.to(self.device)
            # Ensure model is in bfloat16
            self.model.to(torch.bfloat16)

        # CRITICAL FIX for LoRA + Gradient Checkpointing + Zero Loss
        # This ensures gradients are allowed to flow through the frozen base model inputs
        # Always enable this for LoRA to ensure gradients flow
        self.model.enable_input_require_grads()
        if self.training_config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            
        # Re-verify and wrap model with LoRA to ensure parameters are registered correctly
        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            task_type="CAUSAL_LM",
            lora_dropout=self.lora_config['lora_dropout'],
            bias="none"
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Ensure generation config is set for exploration
        if self.model.generation_config:
            self.model.generation_config.do_sample = True
            self.model.generation_config.temperature = 0.9
            self.model.generation_config.top_p = 0.9
            logger.info(f"Updated generation config: do_sample={self.model.generation_config.do_sample}, temp={self.model.generation_config.temperature}")
            
        self.model.print_trainable_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.model_config['cache_dir'],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_reward_functions(self):
        """Define reward functions based on config."""
        reward_type = self.grpo_config.get('reward_function', 'length')
        
        if reward_type == 'length':
            # Reward longer responses (simple proxy for detail)
            def length_reward_func(prompts, completions, **kwargs):
                return [float(len(c)) for c in completions]
            return [length_reward_func]
            
        elif reward_type == 'math_reasoning':
            # GSM8K specific rewards: Soft rewards to bootstrap learning
            
            def extract_answer(text):
                # GSM8K answers are usually "#### <number>"
                if "####" not in text:
                    return None
                return text.split("####")[1].strip()

            def soft_format_reward_func(prompts, completions, **kwargs):
                """Reward for attempting the standard format."""
                rewards = []
                for c in completions:
                    if "####" in c:
                        rewards.append(0.1)
                    else:
                        rewards.append(0.0)
                return rewards

            def strict_format_reward_func(prompts, completions, **kwargs):
                """Reward for using the specific thinking tags."""
                rewards = []
                for c in completions:
                    if "<think>" in c and "</think>" in c:
                        rewards.append(0.5)
                    else:
                        rewards.append(0.0)
                return rewards

            def int_reward_func(prompts, completions, **kwargs):
                """Reward for outputting a valid integer as the answer."""
                rewards = []
                for c in completions:
                    ans = extract_answer(c)
                    if ans and ans.replace(',', '').replace('.', '').isdigit():
                         rewards.append(0.5)
                    else:
                        rewards.append(0.0)
                return rewards

            def correctness_reward_func(prompts, completions, answer, **kwargs):
                """Hard reward for exact correctness."""
                rewards = []
                for completion, correct_answer in zip(completions, answer):
                    gold_val = extract_answer(correct_answer)
                    if not gold_val:
                        gold_val = correct_answer.strip().split()[-1]
                    
                    pred_val = extract_answer(completion)
                    if not pred_val:
                         # Fallback: find last number
                         numbers = re.findall(r'-?\d+\.?\d*', completion)
                         if numbers:
                             pred_val = numbers[-1]
                    
                    # Normalize
                    if pred_val:
                        pred_val = pred_val.replace(',', '').strip()
                        if '.' in pred_val: pred_val = pred_val.split('.')[0]
                    if gold_val:
                        gold_val = gold_val.replace(',', '').strip()
                        if '.' in gold_val: gold_val = gold_val.split('.')[0]

                    if pred_val and gold_val and pred_val == gold_val:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                return rewards

            return [soft_format_reward_func, strict_format_reward_func, int_reward_func, correctness_reward_func]
            
        elif reward_type == 'format':
            # Reward responses that have XML tags (DeepSeek style)
            def xml_format_reward_func(prompts, completions, **kwargs):
                rewards = []
                for c in completions:
                    score = 0.0
                    if "<think>" in c and "</think>" in c:
                        score += 1.0
                    rewards.append(score)
                return rewards
            return [xml_format_reward_func]
        
        elif reward_type == 'accuracy':
            # A dummy accuracy reward that prefers responses not starting with "I don't know"
            def accuracy_reward_func(prompts, completions, **kwargs):
                rewards = []
                for c in completions:
                    if "I don't know" not in c:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                return rewards
            return [accuracy_reward_func]
        
        elif reward_type == 'tool_use':
            # Encourage well-formed function calls when tools are available
            def tool_use_reward_func(prompts, completions, **kwargs):
                rewards = []
                for prompt, completion in zip(prompts, completions):
                    has_tools = '"parameters"' in prompt and '"name"' in prompt
                    uses_tool = "<functioncall>" in completion
                    reward = 0.0
                    if has_tools and not uses_tool:
                        reward -= 0.4
                    elif has_tools and uses_tool:
                        reward += 0.4
                    elif not has_tools and uses_tool:
                        reward -= 0.1
                    else:
                        reward += 0.1
                    if uses_tool:
                        payload_section = completion.split("<functioncall>", 1)[1]
                        payload_section = payload_section.split("<|endoftext|>", 1)[0]
                        json_match = re.search(r'\{.*\}', payload_section, re.DOTALL)
                        if json_match:
                            try:
                                data = json.loads(json_match.group(0))
                                reward += 0.3
                                if isinstance(data, dict) and 'name' in data and 'arguments' in data:
                                    reward += 0.2
                                    if isinstance(data['arguments'], dict):
                                        reward += 0.1
                            except json.JSONDecodeError:
                                reward -= 0.3
                        else:
                            reward -= 0.2
                    rewards.append(float(max(min(reward, 1.5), -1.0)))
                return rewards
            return [tool_use_reward_func]
            
        return []

    def _format_instruction_prompt(self, example):
        instruction = example['instruction']
        input_text = example.get('input', '')
        if input_text:
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )
        return {"prompt": prompt}

    def _format_glaive_prompt(self, example):
        system_prompt = example.get('system', '').strip()
        chat_history = example.get('chat', '')
        assistant_idx = chat_history.find("ASSISTANT:")
        if assistant_idx != -1:
            conversation_up_to_assistant = chat_history[:assistant_idx]
        else:
            conversation_up_to_assistant = chat_history
        prompt_sections = []
        if system_prompt:
            prompt_sections.append(system_prompt)
        if conversation_up_to_assistant.strip():
            prompt_sections.append(conversation_up_to_assistant.strip())
        prompt = "\n\n".join(prompt_sections).rstrip()
        if not prompt:
            prompt = "ASSISTANT:"
        elif not prompt.endswith("ASSISTANT:"):
            prompt = f"{prompt}\nASSISTANT:"
        return {"prompt": prompt}

    def _format_gsm8k_prompt(self, example):
        """Format GSM8K examples. Keep 'answer' for reward calculation."""
        # GSM8K has 'question' and 'answer'
        question = example['question']
        
        # Simple prompt encouraging reasoning
        prompt = (
            "Answer the following math question. "
            "Think step-by-step before giving the final answer.\n"
            "Put your thinking process inside <think> tags and the final answer after ####.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return {"prompt": prompt, "answer": example['answer']}

    def prepare_dataset(self):
        """Prepare dataset for GRPO. GRPO mainly needs 'prompt'."""
        data_prep = DataPreparation()
        
        # We need to manually load specific datasets if not going through data_prep's generic loader
        # or rely on data_prep.load_dataset_from_hub() if it handles config well
        dataset = data_prep.load_dataset_from_hub()
        dataset_name = self.dataset_config.get('name', '').lower()
        
        if "gsm8k" in dataset_name:
            # For GSM8K, we need to map but KEEP the answer column for the reward function
            # GRPOTrainer passes columns present in dataset to reward func if they match args
            formatted_dataset = dataset.map(self._format_gsm8k_prompt)
            # We don't remove columns for GSM8K because we need 'answer'
            # But we should remove 'question' if not needed to save memory
            return formatted_dataset
            
        elif "glaive" in dataset_name:
            formatter = self._format_glaive_prompt
            dataset = dataset.map(formatter, remove_columns=dataset.column_names)
        else:
            formatter = self._format_instruction_prompt
            dataset = dataset.map(formatter, remove_columns=dataset.column_names)
            
        return dataset

    def train(self):
        self.load_model_and_tokenizer()
        dataset = self.prepare_dataset()
        reward_funcs = self.get_reward_functions()
        
        # Configure GRPO
        training_args = GRPOConfig(
            output_dir=os.path.join(self.training_config['output_dir'], "grpo_model"),
            learning_rate=self.grpo_config['learning_rate'],
            max_steps=1000,
            logging_steps=1,
            max_completion_length=self.grpo_config['max_completion_length'],
            max_prompt_length=self.grpo_config['max_prompt_length'],
            num_generations=self.grpo_config['num_generations'],
            beta=self.grpo_config['beta'],
            per_device_train_batch_size=1, # Keep small for Mac
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            save_strategy="steps",
            save_steps=100,
            report_to="tensorboard",
            use_vllm=False, # vLLM not supported on MPS yet easily
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', False),
            temperature=self.config.get('inference', {}).get('temperature', 0.9), # Ensure diversity
        )
        
        # Disable cache for training (required for gradient checkpointing, good practice for LoRA)
        self.model.config.use_cache = False

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting GRPO training...")
        train_result = trainer.train()
        
        logger.info("Saving GRPO model...")
        trainer.save_model(training_args.output_dir)

        # Save training history
        history_path = os.path.join(training_args.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)
        logger.info(f"Training history saved to {history_path}")

        # Generate plots
        self.plot_metrics(trainer.state.log_history)

    def plot_metrics(self, log_history):
        """Generate plots for GRPO metrics."""
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        logger.info("Generating GRPO metrics plots...")
        df = pd.DataFrame(log_history)
        
        # Filter for rows that have the metrics we want (logging steps)
        # In GRPO, it logs 'loss', 'reward', 'reward_std', 'kl'
        # Note: Depending on the TRL version, reward might be 'rewards/chosen' etc.
        # But for GRPOTrainer it's usually just 'reward' or 'rewards/all'
        
        metrics_to_plot = ['loss', 'reward', 'kl', 'reward_std']
        existing_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if not existing_metrics:
            logger.warning("No metrics found in log history to plot.")
            return

        os.makedirs("outputs/plots", exist_ok=True)
        
        # Create a combined plot
        fig, axes = plt.subplots(len(existing_metrics), 1, figsize=(10, 4 * len(existing_metrics)))
        if len(existing_metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(existing_metrics):
            metric_df = df[['step', metric]].dropna()
            sns.lineplot(data=metric_df, x='step', y=metric, ax=axes[i])
            axes[i].set_title(f"GRPO {metric.capitalize()} over Steps")
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel(metric)
        
        plt.tight_layout()
        combined_plot_path = "outputs/plots/grpo_metrics_combined.png"
        plt.savefig(combined_plot_path)
        logger.info(f"Combined metrics plot saved to {combined_plot_path}")
        
        # Also save individual plots
        for metric in existing_metrics:
            plt.figure(figsize=(8, 5))
            metric_df = df[['step', metric]].dropna()
            sns.lineplot(data=metric_df, x='step', y=metric)
            plt.title(f"GRPO {metric.capitalize()} over Steps")
            plt.savefig(f"outputs/plots/grpo_{metric}.png")
            plt.close()

if __name__ == "__main__":
    pipeline = GRPOTrainingPipeline()
    pipeline.train()
