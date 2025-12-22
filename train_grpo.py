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
            'torch_dtype': torch.float16,  # Use float16 on Mac
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

    def prepare_dataset(self):
        """Prepare dataset for GRPO. GRPO mainly needs 'prompt'."""
        data_prep = DataPreparation()
        dataset = data_prep.load_dataset_from_hub()
        dataset_name = self.dataset_config.get('name', '').lower()
        if "glaive" in dataset_name:
            formatter = self._format_glaive_prompt
        else:
            formatter = self._format_instruction_prompt
        dataset = dataset.map(formatter, remove_columns=dataset.column_names)
        return dataset

    def train(self):
        self.load_model_and_tokenizer()
        dataset = self.prepare_dataset()
        reward_funcs = self.get_reward_functions()
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            task_type="CAUSAL_LM",
            lora_dropout=self.lora_config['lora_dropout'],
            bias="none"
        )

        # Configure GRPO
        training_args = GRPOConfig(
            output_dir=os.path.join(self.training_config['output_dir'], "grpo_model"),
            learning_rate=self.grpo_config['learning_rate'],
            num_train_epochs=1, # Usually 1 epoch for RL steps
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
            use_vllm=False # vLLM not supported on MPS yet easily
        )

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=self.tokenizer,
        )

        logger.info("Starting GRPO training...")
        trainer.train()
        
        logger.info("Saving GRPO model...")
        trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    pipeline = GRPOTrainingPipeline()
    pipeline.train()
