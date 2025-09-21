import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json

def filter_dataset(data, skip_questions=None):
    """Filter out unwanted questions from the dataset"""
    if skip_questions is None:
        skip_questions = ["What is mentioned in the claim?"]
    
    filtered_data = []
    skipped_count = 0
    
    print("Filtering dataset...")
    for item in tqdm(data, desc="Processing samples"):
        # Handle different field names for question
        question_key = 'question' if 'question' in item else 'src'
        question = item[question_key].strip()
        
        # Check if question should be skipped
        should_skip = False
        for skip_q in skip_questions:
            if question.lower() == skip_q.lower():
                should_skip = True
                skipped_count += 1
                break
        
        if not should_skip:
            filtered_data.append(item)
    
    print(f"Filtered dataset: {len(filtered_data)} samples kept, {skipped_count} samples skipped")
    return filtered_data

class QADataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        # Filter out unwanted questions before creating dataset
        self.data = filter_dataset(data)
        self.max_length = max_length
        
        # Pre-tokenize all data for faster training
        print("Pre-tokenizing dataset...")
        self.tokenized_data = []
        for item in tqdm(self.data, desc="Tokenizing samples"):
            # Handle different field names for question and answer
            question_key = 'question' if 'question' in item else 'src'
            answer_key = 'answer' if 'answer' in item else 'pred'
            
            # Format the prompt
            prompt = f"""You are a helpful assistant, answer the questions below.

### Question:
{item[question_key]}

### Answer:
{item[answer_key]}{self.tokenizer.eos_token}"""
            
            # Tokenize
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            self.tokenized_data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()
            })
        
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def setup_model_and_tokenizer(model_name, use_flash_attention=False):
    """Load model and tokenizer with proper configuration"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Model configuration for full fine-tuning
    model_kwargs = {
        "torch_dtype": torch.bfloat16,  # Use bfloat16 for better stability
        "device_map": "auto",           # Automatically distribute across GPUs
        "trust_remote_code": True,
    }
    
    # Add flash attention if requested (requires flash-attn package)
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Resize token embeddings if tokenizer was modified
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

def create_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    """Create and configure the trainer"""
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        return_tensors="pt",
        pad_to_multiple_of=8,  # Optimize for tensor cores
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./outputs/{args.model_save_name}",
        overwrite_output_dir=True,
        
        # Training hyperparameters
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        
        # Batch sizes and gradient accumulation
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Memory optimizations
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        optim="adamw_torch",  # or "adamw_8bit" for 8-bit optimizer
        
        # Logging and evaluation
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        
        # Enhanced progress tracking
        disable_tqdm=False,  # Keep tqdm progress bars
        log_level="info",    # More detailed logging
        logging_first_step=True,  # Log the first training step
        
        # Mixed precision training
        fp16=False,  # Set to True if using older GPUs
        bf16=True,   # Better for newer GPUs (A100, H100, etc.)
        
        # Other settings
        remove_unused_columns=False,
        report_to="none",  # Change to "wandb" or "tensorboard" if you want logging
        seed=args.seed,
        
        # Push to hub settings
        push_to_hub=args.push_to_hub,
        hub_model_id=f"{args.hub_username}/{args.model_save_name}" if args.push_to_hub else None,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return trainer

def test_inference(model, tokenizer, device="cuda"):
    """Test the fine-tuned model"""
    model.eval()
    
    test_prompt = """You are a helpful assistant, answer the questions below.

### Question:
What company made USS Leedstown (APA-56)?

### Answer:
"""
    
    # Tokenize input
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 50)
    print("INFERENCE TEST")
    print("=" * 50)
    print(response)
    print("=" * 50)
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Full SFT Training Pipeline")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Base model name from HuggingFace")
    parser.add_argument("--model_save_name", type=str, required=True, help="Name for the fine-tuned model")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (HuggingFace Hub) or path to local .json/.jsonl file")
    parser.add_argument("--train_split", type=str, default="[:8000]", help="Training split (HF format) or slice for local files [start:end]")
    parser.add_argument("--eval_split", type=str, default="[8000:10000]", help="Evaluation split (HF format) or slice for local files [start:end]")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--skip_questions", nargs="*", default=["What is mentioned in the claim?"], 
                       help="Questions to skip during training")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Batch size and memory arguments
    parser.add_argument("--batch_size", type=int, default=2, help="Per device training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Per device evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    
    # Logging and saving arguments
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    
    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_username", type=str, help="HuggingFace Hub username")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_inference_test", action="store_true", help="Run inference test after training")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("FULL SFT TRAINING PIPELINE")
    print("=" * 50)
    print(f"Base Model: {args.model_name}")
    print(f"Save Name: {args.model_save_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Questions to skip: {args.skip_questions}")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load dataset
    print("Loading dataset...")
    
    # Check if dataset_name is a local file or HuggingFace dataset
    if os.path.exists(args.dataset_name):
        print(f"Loading local file: {args.dataset_name}")
        
        if args.dataset_name.endswith('.json') or args.dataset_name.endswith('.jsonl'):
            # Load JSON or JSONL file
            data = []
            with open(args.dataset_name, 'r', encoding='utf-8') as f:
                if args.dataset_name.endswith('.jsonl'):
                    # JSONL format - one JSON object per line
                    print("Loading JSONL file...")
                    lines = f.readlines()
                    for line in tqdm(lines, desc="Reading JSONL lines"):
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                else:
                    # Regular JSON format
                    print("Loading JSON file...")
                    file_content = json.load(f)
                    if isinstance(file_content, list):
                        data = file_content
                    else:
                        data = [file_content]
            
            # Convert to list if it's a dictionary (for regular JSON files)
            if isinstance(data, dict):
                if 'data' in data:
                    data = data['data']
                elif 'examples' in data:
                    data = data['examples']
                else:
                    # If it's a dict with other keys, convert to list of dicts
                    data = [data] if not any(isinstance(v, list) for v in data.values()) else list(data.values())[0]
            
            # Parse split specifications for local files
            total_len = len(data)
            
            # Parse train split (e.g., "[:8000]" or "[0:8000]")
            if args.train_split:
                if args.train_split.startswith('[') and args.train_split.endswith(']'):
                    split_str = args.train_split[1:-1]  # Remove brackets
                    if ':' in split_str:
                        start, end = split_str.split(':')
                        start = int(start) if start else 0
                        end = int(end) if end else total_len
                    else:
                        start = 0
                        end = int(split_str)
                    train_data = data[start:end]
                else:
                    # If no brackets, treat as number of samples from beginning
                    num_samples = int(args.train_split)
                    train_data = data[:num_samples]
            else:
                train_data = data
            
            # Parse eval split
            if args.eval_split:
                if args.eval_split.startswith('[') and args.eval_split.endswith(']'):
                    split_str = args.eval_split[1:-1]
                    if ':' in split_str:
                        start, end = split_str.split(':')
                        start = int(start) if start else 0
                        end = int(end) if end else total_len
                    else:
                        start = int(split_str)
                        end = total_len
                    eval_data = data[start:end]
                else:
                    num_samples = int(args.eval_split)
                    eval_data = data[-num_samples:]  # Take from end
            else:
                eval_data = None
                
        else:
            raise ValueError(f"Unsupported file format: {args.dataset_name}. Only .json and .jsonl files are supported for local files.")
            
    else:
        # Load from HuggingFace Hub
        print(f"Loading from HuggingFace Hub: {args.dataset_name}")
        train_data = load_dataset(args.dataset_name, split=args.train_split)
        eval_data = load_dataset(args.dataset_name, split=args.eval_split) if args.eval_split else None
    
    print(f"Training samples: {len(train_data)}")
    if eval_data:
        print(f"Evaluation samples: {len(eval_data)}")
    
    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.use_flash_attention)
    
    # Create datasets
    print("Preparing datasets...")
    train_dataset = QADataset(tokenizer, train_data, args.max_seq_length)
    eval_dataset = QADataset(tokenizer, eval_data, args.max_seq_length) if eval_data else None
    
    # Create trainer
    print("Setting up trainer...")
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, args)
    
    # Start training
    print("Starting training...")
    train_result = trainer.train()
    
    # Print training results
    print("=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Training Loss: {train_result.training_loss:.4f}")
    print(f"Training Steps: {train_result.global_step}")
    
    # Save model
    print("Saving model...")
    trainer.save_model()
    trainer.save_state()
    
    # Run inference test
    if args.run_inference_test:
        print("Running inference test...")
        test_inference(model, tokenizer)
    
    # Push to hub if requested
    if args.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        trainer.push_to_hub(commit_message="Fine-tuned with full SFT")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()