"""
Training Script for SFT (Supervised Fine-Tuning) of LLMs
Optimized for Mac M1 with Apple Silicon support
"""

import os
import yaml
import torch
import logging

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Try to import BitsAndBytesConfig, but it's not available on macOS
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes not available on macOS - using float16 precision instead")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from data_preparation import DataPreparation


class LLMTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the LLM trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Use torch.device objects (not raw strings) for safe device handling
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        logger.info(f"Using device: {self.device}")

        self.model = None
        self.tokenizer = None
        self.data_prep = DataPreparation(config_path)

    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with quantization."""
        model_config = self.config['model']
        quant_config = self.config['quantization']

        logger.info(f"Loading model: {model_config['name']}")

        # Setup quantization config for memory efficiency
        # Note: bitsandbytes not available on macOS, use dtype instead
        bnb_config = None
        load_kwargs = {
            'cache_dir': model_config['cache_dir'],
            'trust_remote_code': True,
            'token': model_config.get('use_auth_token'),
            'device_map': 'auto'
        }

        # If running on MPS, avoid device_map='auto' because it may produce meta tensors
        is_mps = (self.device.type == 'mps')
        if is_mps:
            logger.info("Detected MPS device - using safe load path (no device_map)")
            # remove device_map so from_pretrained materializes parameters normally
            load_kwargs.pop('device_map', None)
            # prefer float32 on MPS for stability; user can change if they want more memory saving
            load_kwargs['dtype'] = torch.float32
            # Set low_cpu_mem_usage=False to avoid meta tensors on MPS
            load_kwargs['low_cpu_mem_usage'] = False

        if BITSANDBYTES_AVAILABLE and quant_config['load_in_4bit'] and not is_mps:
            logger.info("Using 4-bit quantization with bitsandbytes")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
                bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant']
            )
            load_kwargs['quantization_config'] = bnb_config
        else:
            # macOS or no bitsandbytes: Use float16 for non-MPS, float32 for MPS
            if is_mps:
                logger.info("Using float32 on MPS for numerical stability")
            else:
                logger.info("Using float16 precision (bitsandbytes not available on macOS)")
                load_kwargs['dtype'] = torch.float16

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            cache_dir=model_config['cache_dir'],
            trust_remote_code=True,
            token=model_config.get('use_auth_token')
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        # If using MPS we intentionally avoid device_map to prevent meta tensors
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['name'],
            **load_kwargs
        )

        # If MPS, move model to mps device explicitly
        if is_mps:
            try:
                self.model.to(self.device)
            except Exception:
                logger.warning("Failed to move model to MPS device; continuing with CPU device placement")

        # Enable gradient checkpointing only when safe
        # Gradient checkpointing has known issues on MPS; disable it to avoid invalid gradient devices
        if self.config['training']['gradient_checkpointing']:
            if is_mps:
                logger.warning("Disabling gradient_checkpointing on MPS due to known backward device/gradient issues")
                self.config['training']['gradient_checkpointing'] = False
            else:
                self.model.gradient_checkpointing_enable()

        logger.info("Model and tokenizer loaded successfully")

    def setup_peft(self):
        """Setup PEFT (Parameter-Efficient Fine-Tuning) with LoRA."""
        lora_config = self.config['lora']

        logger.info("Setting up LoRA configuration")

        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type=lora_config['task_type'],
            target_modules=lora_config['target_modules']
        )

        # Prepare model for k-bit training if using quantization with bitsandbytes
        if BITSANDBYTES_AVAILABLE and self.config['quantization']['load_in_4bit']:
            logger.info("Preparing model for k-bit training")
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            logger.info("Using standard LoRA training (no quantization)")

        # Get PEFT model
        self.model = get_peft_model(self.model, peft_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def tokenize_function(self, examples):
        """Tokenize the text data."""
        result = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config['training']['max_seq_length'],
            padding="max_length"
        )

        # Set labels for causal language modeling
        result["labels"] = result["input_ids"].copy()

        return result

    def prepare_datasets(self):
        """Prepare training and evaluation datasets."""
        logger.info("Preparing datasets")

        train_dataset, eval_dataset = self.data_prep.prepare_dataset(
            format_style="alpaca"
        )

        # Tokenize datasets
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        eval_dataset = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        return train_dataset, eval_dataset

    def train(self):
        """Run the training loop."""
        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Setup PEFT
        self.setup_peft()

        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets()

        # Training arguments
        training_config = self.config['training']
        logging_config = self.config['logging']

        # Common kwargs for TrainingArguments (exclude eval param to handle compatibility)
        common_training_kwargs = dict(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            warmup_ratio=training_config['warmup_ratio'],
            weight_decay=training_config['weight_decay'],
            max_grad_norm=training_config['max_grad_norm'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            save_total_limit=training_config['save_total_limit'],
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],
            optim=training_config['optim'],
            gradient_checkpointing=training_config['gradient_checkpointing'],
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to=logging_config['report_to'],
            run_name=logging_config.get('wandb_run_name'),
            logging_dir=f"{training_config['output_dir']}/logs",
        )

        # Create TrainingArguments with compatibility for different HF versions
        try:
            # Preferred param name in newer transformers
            training_args = TrainingArguments(**common_training_kwargs, eval_strategy="steps")
        except TypeError:
            # Fallback for older transformers
            training_args = TrainingArguments(**common_training_kwargs, evaluation_strategy="steps")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal LM, not masked LM
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        logger.info("Training complete! Saving model...")
        trainer.save_model(f"{training_config['output_dir']}/final_model")
        self.tokenizer.save_pretrained(f"{training_config['output_dir']}/final_model")

        logger.info(f"Model saved to {training_config['output_dir']}/final_model")

        return trainer


def main():
    """Main training function."""
    trainer = LLMTrainer(config_path="config.yaml")
    trainer.train()


if __name__ == "__main__":
    main()
