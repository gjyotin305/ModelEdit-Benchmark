import os
import json
import yaml
import argparse
import torch
from tqdm import tqdm
from transformers import GenerationConfig
from utils_expert import get_model_arbitrary, is_correct_new_schema


generation_config = GenerationConfig(
    num_beams=1,
    max_new_tokens=20,
    min_new_tokens=1,
    repetition_penalty=1.1,
    do_sample=True,
    top_k=50,
)

zsre_prompt_q = """
You are a helpful assistant, answer the questions below.

### Question:
{}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="editing_techniques/MAAT/config_unsloth.yml", help="Path to config YAML")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to evaluate from the dataset start")
    parser.add_argument("--model", type=str, default=None, help="Override model id (defaults to config zsRE_edit_model)")
    parser.add_argument("--data", type=str, default=None, help="Override dataset path (defaults to config zsRE_edit_data)")
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU id (defaults to config gpus)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_id = args.model if args.model is not None else config["zsRE_edit_model"]
    data_path = args.data if args.data is not None else config["zsRE_edit_data"]
    gpu_id = args.gpu if args.gpu is not None else config["gpus"]

    model, tokenizer = get_model_arbitrary(model_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for _n, _p in model.named_parameters():
        _p.requires_grad = False

    with open(data_path, "r") as f:
        dataset = json.load(f)

    total = min(args.samples, len(dataset))
    print(f"Running baseline inference on {total} samples using {model_id} (device: {device})")

    correct = 0
    for idx, item in tqdm(enumerate(dataset[:total])):
        prompt = zsre_prompt_q.format(item["src"]) 
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(input_ids, generation_config=generation_config)
        # Trim prompt tokens from the output, then decode only generated part
        gen_only_ids = output_ids[0, input_ids.shape[-1]:]
        response = tokenizer.decode(gen_only_ids, skip_special_tokens=True)

        # Pretty print question, predicted response, and expected answer
        print(
            f"Question:"
            f"  {item['src']}\n"
            "Predicted:"
            f"  {response.strip()}\n"
            "Expected:"
            f"  {item['pred']}\n"
        )

        if is_correct_new_schema(response, item["pred"], tokenizer.eos_token):
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy@pred (first {total}): {acc:.4f}")


if __name__ == "__main__":
    main()


