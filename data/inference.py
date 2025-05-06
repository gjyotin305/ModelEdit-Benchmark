from vllm import LLM, SamplingParams
from typing import List, Dict
from tqdm import tqdm
import json

llm = LLM(model="microsoft/Phi-4-mini-instruct", trust_remote_code=True)
sampling_params = SamplingParams(
    max_tokens=1000,
    temperature=0.0
)


def create_batch_push(batch_size: int, data: List[Dict]) -> None:
    length = len(data)
    results = []

    print("Total Length of claims/evidences are :", len(data))

    for i in tqdm(range(0, 50000, batch_size)):
        print(f'Creating a batch of {batch_size}')
        
        if i+batch_size <= length:
            temp_data = data[i: i+batch_size]
        else:
            temp_data = data[i:]

        assert len(temp_data) == batch_size

        try:
	        output = create_dataset_object(temp_data)
        except Exception as e:
            print(f"Exception Occured {e} | Index {i}")
            continue
         
        for id, out in enumerate(output[:batch_size]):
            results.append(
                {
                    "output": out.outputs[0].text,
                    "5W_id": temp_data[id]['id'] 
                }
            )
        
        with open("/scratch/data/asif_rs/data/5WQA_Edit_dataset_v1.json", 'w') as f:
            json.dump(results, f)
            f.close()

    print(f"Objects received are: {len(results)}")

def create_dataset_object(object: List[str]):
    chat_template = """<|system|>You will be given a passage of text. Your task is to generate five questions based on the "5Ws" framework: What, Where, When, Why, and Who. Each question should be relevant to the information provided in the text and should focus on concepts, locations, events, or key details.For each question, also provide a corresponding answer based on the text. Ensure the output is structured in a valid JSON format, with each question-answer pair properly formatted.<|end|>  
    <|user|>TEXT: {}<|end|>  
    <|assistant|>  
    """

    prompts = [chat_template.format(text['evidence']) for text in object]

    print(f"Batch Received of {len(prompts)}")

    output = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params
    )

    return output


if __name__ == "__main__":
    with open("/scratch/data/asif_rs/data/5WQA_all_claims_with_evidence.json", 'r') as f:
        data_ = json.load(f)
        f.close()

    print(len(data_))
    
    create_batch_push(batch_size=16, data=data_)
