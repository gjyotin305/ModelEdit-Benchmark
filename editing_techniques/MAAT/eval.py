from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm
import json

def jsonl_to_json(jsonl_path, json_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

# test_ds = load_dataset("gjyotin305/full_zsre", split="test[10000:]")
# test_ds.to_json('temp.jsonl')

zsre_prompt = """
You are a helpful assistant, answer the questions below.

### Question:
{}

### Answer:
{}"""

# jsonl_to_json('temp.jsonl', './Data/test_zsre.json')

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

def evaluate_pair(model, tokenizer, src, gt):
    inputs = tokenizer([
        zsre_prompt.format(src, "")
    ], return_tensors="pt").to('cuda')

    outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
    results = tokenizer.batch_decode(outputs)
    # print(results[0])
    # print(gt)
    prediction = results[0].split("### Answer:")[-1].split(tokenizer.eos_token)[0].strip()
    # print(prediction)
    if prediction == gt:
        return True
    else:
        return False
    
if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="gjyotin305/qwen3_zsre_merged",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=False
    )

    with open('./Data/test_zsre.json', 'r') as f:
        data = json.load(f)

    score = []
    for point in tqdm(data[:5000], total=len(data[:5000]), desc="Evaluating"):
        result = evaluate_pair(model, tokenizer, point['src'], point['pred'])
        point['qwen_result'] = result
        score.append(result)
    
    print("="*10 + " Final Score " + "="*10)
    print(f"Final Accuracy: {(sum(score)/len(score))*100}")
    with open('./Data/test_zsre_qwen_locality.json', 'w') as f:
        json.dump(data, f)
    
    
