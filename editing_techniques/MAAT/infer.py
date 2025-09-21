import os
import json
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from utils_expert import extract_number,is_correct,get_model,parent_module, get_model_arbitrary, is_correct_new_schema
from transformers import GenerationConfig

generation_config = GenerationConfig(
    num_beams=1,
    max_new_tokens=100,  
    min_new_tokens=1,
    repetition_penalty=1.1,
    do_sample=False,
)

zsre_prompt_q = """
You are a helpful assistant, answer the questions below.

### Question:
{}"""

zsre_prompt_a = """

### Answer:
{}"""

def merge_neruans(config):

    # merge indexing neurals
    path = config["neruals_save_path"] + config["lab_tag"]+ "/" + config["modify_layer_names"][0] + "/"
    file_names = os.listdir(path)
    file_names = sorted(file_names, key=extract_number)
    # print(file_names)
    all_tmp_weight = []
    assert len(file_names) == config["seq_length"]
    for file in file_names[:]:
        tmp_weight = torch.load(path + file).cpu()
        all_tmp_weight.append(tmp_weight)
    weights_matrix = torch.cat(all_tmp_weight, dim=0) 

    fc_layer = nn.Linear(config["mid_hidden"], len(file_names), bias=False)
    fc_layer.weight.data = weights_matrix
    torch.save(fc_layer.weight, config["merge_neruals"] + config["lab_tag"] +"_indexing_neurons.pt")


def main():
    with open("editing_techniques/MAAT/config_unsloth_5wqa.yml","r") as f:
        config = yaml.safe_load(f)

    class SCEN_select_expert(nn.Module):
        def __init__(self,layer, router_path = "",experts_fold= "",experts = config["seq_length"], flag_first = 1,):
            super(SCEN_select_expert, self).__init__()
            self.layer = layer
            self.experts_fload = experts_fold
            self.expert_learning = nn.Linear(layer.in_features, layer.out_features, bias=False)   
            self.router = nn.Linear(layer.in_features, experts, bias=False)

            self.flag_first = flag_first #flag = 1 first check

            # load router weight 
            self.router.weight = torch.load(router_path)
            
        def forward(self, *args):
            if self.flag_first == 1:
                self.router_res = torch.sigmoid(self.router(*args))

                # ADD THE DEBUG PRINTS HERE:
                print(f"All router confidences: {self.router_res[0,-1,:].tolist()}")
                print(f"Expert indices: {list(range(len(self.router_res[0,-1,:].tolist())))}")
                
                max_val_index = torch.argmax(self.router_res[0,-1,:]).tolist()
                max_val = torch.max(self.router_res[0,-1,:]).tolist()


                # ADD THESE DEBUG PRINTS:
                print(f"Router confidence: {max_val:.4f}, Threshold: {config['theta']}")
                print(f"Selected expert: {max_val_index}, Router shape: {self.router_res.shape}")
        
                if max_val>config["theta"]: 
                    print(f"✓ ACTIVATING EXPERT {max_val_index+1}")
                    # forward expert weight so load expert weight
                    self.expert_learning.weight = torch.load(self.experts_fload+"expert_learning_weight{}.pt".format(max_val_index+1))
                    self.expert_learning.to(self.router_res.device) 
                    self.flag_first =2
                    return self.expert_learning(*args)
                else:
                    print(f"✗ USING ORIGINAL MODEL (confidence {max_val:.4f} < threshold {config['theta']})")
                    # forward raw weight 
                    self.flag_first =3
                    return  self.layer(*args)
            elif self.flag_first == 2:
                return self.expert_learning(*args)
            elif self.flag_first == 3:
                return  self.layer(*args)

    # merge then save
    merge_neruans(config)
    print('=======================merge done=======================')

    # start infer  
    router_name = config["lab_tag"] + "_indexing_neurons.pt"
    weight_path = config["merge_neruals"] + router_name
    experts_fold =  config["expert_save_path"] + config["lab_tag"] +"/"+config["modify_layer_names"][0] +"/"

    edit_layer_name = config["modify_layer_names"][0]
    modify_layer_names = [
    edit_layer_name
    ]
    gpu_id = config["gpus"]

    model,tokenizer = get_model_arbitrary(config["zsRE_edit_model"])
    device = torch.device(f"cuda:{gpu_id}")
    model.to(device)

    for n, p in model.named_parameters():
        p.requires_grad = False

    with open(config["zsRE_edit_data"], "r") as f:
        edit_data = json.load(f)

    # with open(config["zsRE_forget_data"], "r") as f:
    #     forget_data = json.load(f)

    print('=======================begin infer=======================')

    original_layer_list = []

    for id_pname,pname in enumerate(modify_layer_names):
        parent = parent_module(model, pname)
        layer_name = pname.split(".")[-1]
        if len(original_layer_list)!=len(modify_layer_names):
            original_layer = getattr(parent, layer_name)
            original_layer_list.append(original_layer)
            setattr(parent, layer_name,SCEN_select_expert(original_layer,router_path = weight_path,experts_fold= experts_fold).to(device))
        else:
            setattr(parent, layer_name,SCEN_select_expert(original_layer_list[id_pname],router_path = weight_path,experts_fold= experts_fold).to(device))

    all_res = {}

    edit_success = 0
    count = 0
    for index_train,train_item in tqdm(enumerate(edit_data[1:config["seq_length"]])):
        count += 1
        # learning_prompt = zsre_prompt_q.format(train_item['src'])
        # answer_prompt = zsre_prompt_a.format(train_item['alt']) + tokenizer.eos_token
        learning_prompt = zsre_prompt_q.format(train_item['question'])
        answer_prompt = zsre_prompt_a.format(train_item['pred_answer']) + tokenizer.eos_token
        single_input_ids = tokenizer.encode(learning_prompt, return_tensors='pt').to(device)

        for pname in modify_layer_names:
            parent = parent_module(model, pname)
            layer_name = pname.split(".")[-1]
            original_layer = getattr(parent, layer_name)
            original_layer.flag_first = 1

        res = model.generate(single_input_ids, max_new_tokens=100, temperature=0.1, top_k=50, do_sample=True, num_beams=1)
        res_string = tokenizer.decode(res.tolist()[0])
        # print(res_string)
        # print(f"Question : {train_item['src']}")
        # print(f"ANSWER : {train_item['pred']}")
        print(f"Question : {train_item['question']}")
        print(f"ANSWER : {train_item['pred_answer']}")

        for pname in modify_layer_names:
            parent = parent_module(model, pname)
            layer_name = pname.split(".")[-1]
            original_layer = getattr(parent, layer_name)

        # if is_correct_new_schema(res_string, train_item["pred"], tokenizer.eos_token):
        if is_correct_new_schema(res_string, train_item["pred_answer"], tokenizer.eos_token):
            print("######################HIIIIIIIII################################")
            edit_success += 1
        else:
            pass
    print(edit_success)
    print(count)
    print(f"Reliability: {edit_success/count}")

    # edit_success = 0
    # count = 0
    # for index_train,train_item in tqdm(enumerate(forget_data[0:1000])):
    #     count += 1
    #     learning_prompt = zsre_prompt_q.format(train_item['src'])
    #     answer_prompt = zsre_prompt_a.format(train_item['alt']) + tokenizer.eos_token
    #     single_input_ids = tokenizer.encode(learning_prompt, return_tensors='pt').to(device)

    #     for pname in modify_layer_names:
    #         parent = parent_module(model, pname)
    #         layer_name = pname.split(".")[-1]
    #         original_layer = getattr(parent, layer_name)
    #         original_layer.flag_first = 1

    #     res = model.generate(single_input_ids,generation_config=generation_config)
    #     res_string = tokenizer.decode(res.tolist()[0])

    #     for pname in modify_layer_names:
    #         parent = parent_module(model, pname)
    #         layer_name = pname.split(".")[-1]
    #         original_layer = getattr(parent, layer_name)

    #     if is_correct(res_string, train_item["alt"]):
    #         edit_success += 1
    #     else:
    #         pass

    # all_res["Locality"] = edit_success/count


    # edit_success = 0
    # count = 0
    # for index_train,train_item in tqdm(enumerate(edit_data[:config["seq_length"]])):
    #     for rewrite_item in train_item["rephrases"][0:3]:
    #         count += 1
    #         learning_prompt = zsre_prompt_q.format(train_item['src'])
    #         answer_prompt = zsre_prompt_a.format(train_item['alt']) + tokenizer.eos_token
    #         single_input_ids = tokenizer.encode(learning_prompt, return_tensors='pt').to(device)

    #         for pname in modify_layer_names:
    #             parent = parent_module(model, pname)
    #             layer_name = pname.split(".")[-1]
    #             original_layer = getattr(parent, layer_name)
    #             original_layer.flag_first = 1

    #         res = model.generate(single_input_ids,generation_config=generation_config)
    #         res_string = tokenizer.decode(res.tolist()[0])

    #         for pname in modify_layer_names:
    #             parent = parent_module(model, pname)
    #             layer_name = pname.split(".")[-1]
    #             original_layer = getattr(parent, layer_name)

    #         if is_correct(res_string, train_item["alt"]):
    #             edit_success += 1
    #         else:
    #             pass
    # all_res["Generality"] = edit_success/count

    # all_res["avg"] = (all_res["Reliability"]+all_res["Locality"]+all_res["Generality"])/3

    # all_res["lr"] = config["lr"] 
    # all_res["seq_length"] = config["seq_length"]
    # all_res["lab"] = config["lab_tag"]
    # all_res["theta"] = config["theta"]
    # print("results have been saved")
    # with open(config["report_res"]+config["lab_tag"]+".json","a+") as f:
    #     json.dump(all_res,f,ensure_ascii=False)
    #     f.write("\n")
        

if __name__ == '__main__':
    main()