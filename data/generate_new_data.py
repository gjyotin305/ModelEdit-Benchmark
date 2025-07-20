from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import re
import random
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import List
import json

random.seed(42)

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8008/v1"
)

system_prompt_og_answer_why = """
You will be given a claim and an evidence, and along with it a question and answer pairs pertaining to it, your job is to make the answer valid
Take the question and answer pair relevant to the aspect: {}

And make a better and relevant answer and question and give it in the following json format

```json
{
    'question': <new_question>,
    'answer': <new_answer>
}
```
"""

system_prompt_og_answer = """
You will be given a claim and an evidence, and along with it a question and answer pairs pertaining to it, your job is to make the answer valid
Take the QA_PAIR relevant to the aspect given: {}

And make a better question (keeping the aspect as the first word) and give it in the following json format, don't change the answer at all.

```json
{
    'question': <new_question>,
    'answer': <answer>
}
```
"""

system_prompt_pred_answer = """
You will be given a question and answer pair, make a new answer completely different and random, but in the same space as the original answer. Give it in a json format

```json
{
    'answer': <new_answer>
}
```
"""

def make_og_answer_corpus(file_name: str, limit: int = 400):
    with open(file_name, 'r') as f:
        data = json.load(f)

    random_data = random.sample(data, limit+100)
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_og_answer, dp) for dp in random_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing data point: {e}")

    assert len(results) >= limit
    
    with open(f'{file_name.split(".")[0]}_verified.json', 'w') as f:
        json.dump(results, f, indent=2)
    

def make_og_answer(data_point):
    result = {
        'claim': data_point['claim'],
        'evidence': data_point['evidence']
    }

    def process(aspect):
        result_ = refine_og_answer(data_point, aspect)
        return string_to_dict(result_)

    aspects = ['why', 'what', 'who', 'where', 'when']
    
    # Each thread gets one aspect to process
    with ThreadPoolExecutor(max_workers=len(aspects)) as executor:
        pairs = list(executor.map(process, aspects))

    result['qa_pairs'] = pairs
    return result



def make_query_og_answer(claim, evidence, QA_PAIRS, aspect):
    return f"CLAIM: {claim} \n EVIDENCE: {evidence} \n FOCUS_ASPECT: **{aspect}** \n QA_PAIRS: {QA_PAIRS}"

def make_pred_answer(question, answer):
    return f"QUESTION: {question} | ANSWER: {answer}"

def extract_dict_block(text):
    pattern = r"""
        \{[^{}]*                                              # Opening brace and non-nested content
        (?P<qk>['"])question(?P=qk)\s*:\s*(?P<qv>['"]).*?(?P=qv)\s*,\s*  # 'question': '...' (mixed quotes ok)
        (?P<ak>['"])answer(?P=ak)\s*:\s*(?P<av>['"]).*?(?P=av)           # 'answer': "..."
        [^{}]*\}                                              # Closing non-nested content and brace
    """
    match = re.search(pattern, text, re.VERBOSE | re.DOTALL)
    return match.group(0) if match else None


def extract_answer_dict(text):
    pattern = r"""
        \{                            # Opening brace
        \s*(['"])answer\1             # Key: 'answer' or "answer"
        \s*:\s*(['"])(.*?)\2          # Value in matching quotes, captured in group 3
        \s*\}                         # Closing brace
    """
    match = re.search(pattern, text, re.VERBOSE)
    if match:
        return {
            "answer": match.group(3)
        }
    return None


def string_to_dict(text: str) -> dict | None:
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
        else:
            print("Parsed object is not a dictionary.")
            return None
    except Exception as e:
        print(f"Error parsing string to dict: {e}")
        return None

def refine_og_answer(data_point, aspect):
    query = make_query_og_answer(claim=data_point['claim'], evidence=data_point['evidence'], aspect=aspect, QA_PAIRS=data_point['qa_pairs'])
    
    if aspect == 'why':
        response = get_response(client=client, system_prompt=system_prompt_og_answer_why, query=query)
    else:
        response = get_response(client=client, system_prompt=system_prompt_og_answer, query=query)
    # print(response)
    result = extract_dict_block(text=response)

    return result

def classify_question(question: str) -> str:
    first_word = question.strip().split()[0].lower()
    classes = {'who', 'what', 'when', 'where', 'why'}
    return first_word if first_word in classes else 'other'

def refine_pred_answer(data_point):
    qa_pairs = data_point['qa_pairs']
    queries = [make_pred_answer(pair['question'], pair['answer']) for pair in qa_pairs]

    def process(pair, query):
        response = get_response(client=client, system_prompt=system_prompt_pred_answer, query=query)
        response_dict = extract_answer_dict(response)
        label = classify_question(question=pair['question'])

        if response_dict is not None:
            return {
                'question': pair['question'],
                'answer': pair['answer'],
                'pred_answer': response_dict['answer'],
                'label': label
            }
        return None

    responses = []
    with ThreadPoolExecutor(max_workers=len(qa_pairs)) as executor:
        futures = [executor.submit(process, pair, query) for pair, query in zip(qa_pairs, queries)]
        for future in futures:
            result = future.result()
            if result is not None:
                responses.append(result)

    return responses

def make_pred_answer_corpus(file_names: List[str], limit: int = 2500):
    results = []

    def process(data_point):
        response = refine_pred_answer(data_point)
        return response

    for file_name in file_names:
        with open(file_name, 'r') as f:
            data = json.load(f)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process, data_point) for data_point in data]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {file_name}"):
                try:
                    response = future.result()
                    if response is not None:
                        results.extend(response)
                except Exception as e:
                    print(f"Error in corpus {e}")
    
    assert len(results) >= limit
    with open('factify_final_edit_dataset.json', 'w') as f:
        json.dump(results, f, indent=4)

def get_response(
    client: OpenAI, 
    system_prompt: str, 
    query: str, 
    model_name: str = "unsloth/Qwen2.5-14B-Instruct"
):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ],
        temperature=0.1,
        max_tokens=4096
    )

    return response.choices[0].message.content


def process_aspect(data, aspect):
    filtered = []
    for claim in tqdm(data):
        value = claim.get('5W_aspects', {}).get(aspect)
        if value and len(value) > 0:
            filtered.append(claim)

    # Save to file
    with open(f'factify_{aspect}.json', 'w') as f:
        json.dump(filtered, f, indent=4)

    return aspect, len(filtered)


def filter_objects_parallel(data):
    aspects = ['who', 'what', 'why', 'when', 'where']

    results = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_aspect, data, aspect): aspect for aspect in aspects}
        
        for future in as_completed(futures):
            aspect, count = future.result()
            results[aspect] = count

    # Print summary
    for aspect, count in results.items():
        print(f"{aspect} aspect | number of pairs {count}")


if __name__ == "__main__":
    # dataset = load_dataset("Novaspree/5W_Factify_QA", split="train")
    # dataset.to_json('factify.json')

    # make_og_answer_corpus('factify_why.json', limit=400)
    # make_og_answer_corpus('factify_what.json', limit=400)
    # make_og_answer_corpus('factify_where.json', limit=400)
    # make_og_answer_corpus('factify_when.json', limit=400)
    # make_og_answer_corpus('factify_who.json', limit=400)

    # with open('factify_why_verified.json', 'r') as f:
    #     data = json.load(f)


    # results = refine_pred_answer(data[0])

    file_names = ['factify_why_verified.json', 'factify_who_verified.json', 'factify_when_verified.json', 'factify_where_verified.json', 'factify_what_verified.json']
    make_pred_answer_corpus(file_names=file_names)
    # print(data[0])

    # response = refine_og_answer(data[1], 'who')
    # print(response)

    # check = make_og_answer(data[1])
    # print(check['qa_pairs'])
    
    # filter_objects_parallel(data)

    # # Now you can access it like:
    # # for item in data:
    # #     print(item["claim"])


    # print(data[0])