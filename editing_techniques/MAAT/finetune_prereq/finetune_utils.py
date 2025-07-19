import unsloth
import torch
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

train_ds = load_dataset("gjyotin305/full_zsre", split="test[:10000]")

max_seq_length = 512
dtype = None
load_in_4bit = False

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Model name to load from HuggingFace")
parser.add_argument("--model_save_name", type=str, help="Directory name to save the fine-tuned model")
parser.add_argument("--num_epochs", type=int, default=1, help="Directory name to save the fine-tuned model")
parser.add_argument("--learning_rate", type=float,default=2e-4, help="Directory name to save the fine-tuned model")
parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Directory name to save the fine-tuned model")
args = parser.parse_args()

model_name = args.model_name
model_save_name = args.model_save_name

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

zsre_prompt = """
You are a helpful assistant, answer the questions below.

### Question:
{}

### Answer:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompt_func(examples):
    question = examples['src']
    answer = examples['pred']

    texts = []
    for input, output in zip(question, answer):
        text = zsre_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def check_model_inference(model, tokenizer):
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
    [
        zsre_prompt.format(
            "What company made USS Leedstown (APA-56)?", # input Question
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 10, use_cache = True)
    results = tokenizer.batch_decode(outputs)
    print("="*10 + " Inference Post Training " + "="*10)
    print(results[0])
    return results

train_ds = train_ds.map(formatting_prompt_func, batched=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs=args.num_epochs,
        learning_rate = args.learning_rate,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = args.lr_scheduler,
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

check_model_inference(model, tokenizer)

model.save_pretrained(f"{model_save_name}_zsre")
tokenizer.save_pretrained(f"{model_save_name}_zsre")
model.push_to_hub(f"{model_save_name}_zsre")
tokenizer.push_to_hub(f"{model_save_name}_zsre")

# model.push_to_hub_merged(f"gjyotin305/{model_save_name}_zsre_merged", tokenizer, save_method = "merged_16bit")

