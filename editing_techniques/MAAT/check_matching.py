from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('gjyotin305/qwen3_zsre_merged')

zsre_prompt_q = """
You are a helpful assistant, answer the questions below.

### Question:
{}"""

zsre_prompt_a = """

### Answer:
{}""" + tokenizer.eos_token

pattern_string = """

### Answer:
"""

test_prompt = zsre_prompt_q.format('Who are you') + zsre_prompt_a.format('Jyotin Goel')
print("="*20)
print(test_prompt)
print("="*20)
all_texts = [test_prompt, test_prompt, test_prompt]
pattern = torch.tensor(tokenizer.encode(pattern_string))

tensor_main = tokenizer(all_texts, padding="longest", return_tensors="pt")
print(tensor_main['input_ids'].shape)

def find_pattern(tensor, pattern, eos_token_id):
    print(f"Input Tensor: {tensor.shape}")
    
    result = torch.full_like(tensor, -100)
    
    for bn, row in enumerate(tensor):
        start_indice_i = find_sequence(row, pattern)
        eos_indice = (row == eos_token_id).nonzero(as_tuple=True)[0]
        result[bn, start_indice_i:eos_indice+1] = tensor[bn, start_indice_i: eos_indice+1]
    
    return result

def find_sequence(tensor, pattern):
    seq_len = pattern.size(0)
    windows = tensor.unfold(0, seq_len, 1)  # shape: (tensor_len - seq_len + 1, seq_len)
    matches = (windows == pattern).all(dim=1)
    match_indices = matches.nonzero(as_tuple=True)[0]
    return match_indices

check = find_sequence(tensor_main['input_ids'][0], pattern=pattern)
print(check)

check = find_pattern(tensor_main['input_ids'], pattern=pattern, eos_token_id=tokenizer.eos_token_id)
print(check)