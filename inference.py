"""
Inference Script for Fine-tuned LLMs
Load and test your fine-tuned model
"""

import torch
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

# Try to import BitsAndBytesConfig, but it's not available on macOS
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("bitsandbytes not available on macOS - using float16 precision instead")


class LLMInference:
    def __init__(self, base_model_path: str, adapter_path: str = None, config_path: str = "config.yaml"):
        """
        Initialize inference engine.
        
        Args:
            base_model_path: Path to base model or model ID
            adapter_path: Path to LoRA adapter weights (optional)
            config_path: Path to config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Use torch.device objects (not raw strings) for safe device handling
        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        print(f"Using device: {self.device}")
        
        self.load_model(base_model_path, adapter_path)
    
    def load_model(self, base_model_path: str, adapter_path: str = None):
        """Load model and tokenizer."""
        print(f"Loading model from: {base_model_path}")
        
        # Get cache dir and token from config
        model_config = self.config.get('model', {})
        cache_dir = model_config.get('cache_dir', './model_cache')
        auth_token = model_config.get('use_auth_token')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=auth_token
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Setup loading arguments
        load_kwargs = {
            'cache_dir': cache_dir,
            'trust_remote_code': True,
            'token': auth_token,
            'device_map': 'auto'
        }

        # Check quantization config
        quant_config = self.config.get('quantization', {})
        is_mps = (self.device.type == 'mps')
        
        # Safe load path for MPS
        if is_mps:
            print("Detected MPS device - using safe load path (no device_map)")
            # remove device_map so from_pretrained materializes parameters normally
            load_kwargs.pop('device_map', None)
            # prefer float32 on MPS for stability; user can change if they want more memory saving
            load_kwargs['dtype'] = torch.float32
            # Set low_cpu_mem_usage=False to avoid meta tensors on MPS
            load_kwargs['low_cpu_mem_usage'] = False
        
        # Handle Quantization
        if BITSANDBYTES_AVAILABLE and quant_config.get('load_in_4bit', False) and not is_mps:
            print("Using 4-bit quantization with bitsandbytes")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True)
            )
            load_kwargs['quantization_config'] = bnb_config
        else:
            # macOS or no bitsandbytes: Use float16 for non-MPS, float32 for MPS
            if is_mps:
                print("Using float32 on MPS for numerical stability")
            else:
                print("Using float16 precision (bitsandbytes not available on macOS)")
                load_kwargs['dtype'] = torch.float16
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **load_kwargs
        )
        
        # If MPS, move model to mps device explicitly
        if is_mps:
            try:
                self.model.to(self.device)
            except Exception as e:
                print(f"Warning: Failed to move model to MPS device: {e}")
        
        # Load adapter if provided
        if adapter_path:
            if os.path.exists(adapter_path):
                print(f"Loading adapter from: {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.model = self.model.merge_and_unload()  # Merge adapter with base model
            else:
                print(f"Warning: Adapter path '{adapter_path}' does not exist.")
                print("Skipping adapter loading and using base model only.")
                print("Tip: If you haven't trained the model yet, run 'python main.py train'.")
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt using template from config."""
        templates = self.config.get('templates', {})
        style = templates.get('active_style', 'alpaca')
        template_config = templates.get(style, {})
        
        # Fallback if config is missing
        if not template_config:
            # Fallback to hardcoded Alpaca style
            if input_text:
                return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

        if input_text:
            return template_config.get('with_input', '').format(
                instruction=instruction, 
                input=input_text
            )
        else:
            return template_config.get('no_input', '').format(
                instruction=instruction
            )
    
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        do_sample: bool = None
    ) -> str:
        """
        Generate response for given instruction.
        
        Args:
            instruction: The instruction/task description
            input_text: Optional context or input
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (False = greedy decoding)
        
        Returns:
            Generated text response
        """
        # Get defaults from config
        inference_config = self.config.get('inference', {})
        
        # Use provided values or defaults from config or fallback defaults
        max_new_tokens = max_new_tokens if max_new_tokens is not None else inference_config.get('max_new_tokens', 256)
        temperature = temperature if temperature is not None else inference_config.get('temperature', 0.7)
        top_p = top_p if top_p is not None else inference_config.get('top_p', 0.9)
        top_k = top_k if top_k is not None else inference_config.get('top_k', 50)
        do_sample = do_sample if do_sample is not None else inference_config.get('do_sample', True)

        prompt = self.format_prompt(instruction, input_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part. 
        # Note: This simple splitting depends on the specific template structure. 
        # Ideally, special tokens should be used to separate response.
        # For now, we try to split by the last prompt line if possible.
        
        # Attempt to remove the prompt from the start
        if generated_text.startswith(prompt):
             response = generated_text[len(prompt):].strip()
        elif "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1].strip()
        else:
            response = generated_text  # Fallback: return everything if we can't cleanly split

        return response
    
    def chat(self):
        """Interactive chat interface."""
        print("\n=== LLM Chat Interface ===")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to start a new conversation\n")
        
        while True:
            try:
                instruction = input("You: ").strip()
                
                if instruction.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if instruction.lower() == 'clear':
                    print("Conversation cleared.\n")
                    continue
                
                if not instruction:
                    continue
                
                print("Assistant: ", end="", flush=True)
                response = self.generate(
                    instruction=instruction,
                    max_new_tokens=512,
                    temperature=0.7
                )
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned LLM")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./outputs/final_model",
        help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Single instruction to run (if not provided, starts chat mode)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input context for the instruction"
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = LLMInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path
    )
    
    # Run single instruction or start chat
    if args.instruction:
        response = inference.generate(
            instruction=args.instruction,
            input_text=args.input
        )
        print(f"\nInstruction: {args.instruction}")
        if args.input:
            print(f"Input: {args.input}")
        print(f"\nResponse: {response}\n")
    else:
        inference.chat()


if __name__ == "__main__":
    main()

