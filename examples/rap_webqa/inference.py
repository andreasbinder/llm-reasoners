import pickle
from typing import Type, Callable, Optional, Literal, List

import numpy as np
from datasets import load_dataset
from reasoners.visualization import TreeLog
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation

from world_model import WebQAWorldModel, GSM8kState, WebQAAction
from search_config import GSM8kConfig
import utils

import wandb
wandb.init()



def log_hyparams_with_path(json_data, path=[]):
    # If json_data is a dictionary
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            # Append the current key to the path
            new_path = path + [key]
            # If the current key is 'hyparams', log its content and the path
            if key == 'hyparams':
                print(f"Path: {' -> '.join(new_path)}, hyparams: {value}")
            # Recursively call the function for nested dictionaries or lists
            log_hyparams_with_path(value, new_path)
    # If json_data is a list, iterate through its elements
    elif isinstance(json_data, list):
        for index, item in enumerate(json_data):
            # Append the index to the path for list elements
            new_path = path + [str(index)]
            log_hyparams_with_path(item, new_path)


def node_visualizer(x: MCTSNode[GSM8kState, WebQAAction]):
    if not x.state:
        return {}
    if x.state[-1].state_type == "ANSWER":
        return {
            "question": x.state[-1].main_question, 
            "answer": x.state[-1].main_answer
            }
    elif x.state[-1].state_type == "RETRIEVE":
        #, "retrieved_snippets": x.state[-1].retrieved_snippets
        return {
            "context": x.state[-1].context,
            
            #"snippet_id": [snippet_id.split("_")[-1] for snippet_id in x.state[-1].retrieved_sources],
            #"snippet_id": x.state[-1].retrieved_sources,
            "source_id": x.state[-1].retrieved_sources,
            "mod": x.state[-1].modalities,
            #"flags": x.state[-1].flags, 
            "is_gold": x.state[-1].is_gold, 
            "relevance_scores": [f'{relevance_score:.3f}'  for relevance_score in x.state[-1].relevance_scores],
            } 
    elif x.state[-1].state_type == "HYPOTHESIS":
        #, "retrieved_snippets": x.state[-1].retrieved_snippets
        return {
            "proposition": x.state[-1].proposition,    
            "comment": x.state[-1].comment    
            } 
    elif x.state[-1].state_type == "INVALID":
        return {
            "content": "INVALID"
            } 


def rap_webqa(base_model: LanguageModel,
              interactive_prompt: dict,
              useful_prompt: dict,
              search_algo: Type[SearchAlgorithm] = MCTS,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              depth_limit: int = 5,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 2,
              temperature: float = 0.8,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              reward_confidence_default: float = 0.8,
              cum_reward: Callable[[List[float]], float] = np.mean,
              calc_q: Callable[[List[float]], float] = max,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              output_trace_in_each_iter: bool = True,
              aggregate: bool = True,
              path_to_webqa: str = None, # TODO
              split: str = "test", # TODO
              **search_algo_params):
    if not disable_log:
        time_prefix = datetime.now().strftime("%m%d%Y-%H%M%S")
        if log_dir is None:
            log_dir = f'logs/webqa_{search_algo.__name__}/{time_prefix}'
        #os.makedirs(log_dir, exist_ok=resume > 0)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, 'disable_tqdm': disable_tqdm,
                           'output_trace_in_each_iter': output_trace_in_each_iter}
    world_model = WebQAWorldModel(base_model=base_model, prompt=interactive_prompt,
                                  n_confidence=n_confidence, batch_size=batch_size, temperature=temperature,
                                  early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = GSM8kConfig(base_model=base_model, prompt=interactive_prompt, useful_prompt=useful_prompt,
                         n_actions=n_action, batch_size=batch_size, temperature=temperature,
                         reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    if aggregate:
        aggregator = MCTSAggregation(utils.retrieve_answer, weight_policy='edge_inverse_depth')
    else:
        aggregator = None

    dataset = utils.load_webqa_dataset(path_to_webqa, split, resume)
    
    retrieve_hyparams = interactive_prompt["actions"]["RETRIEVE"]["hyparams"]
    checkpoint_embedding_model = retrieve_hyparams["embedding_model"].get("checkpoint", "sentence-transformers/all-mpnet-base-v2")
    from models.mpnet_model import MPNetEmbedder
    #checkpoint_embedding_model = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = MPNetEmbedder(checkpoint_embedding_model)

    #from models.clip_model import CLIP
    #checkpoint_embedding_model = "sentence-transformers/all-mpnet-base-v2"
    #embedding_model = CLIP("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    if retrieve_hyparams['use_caption_model']:

        from models.llava_model import LLAVAModel
        #model_path = "liuhaotian/llava-v1.5-13b"
        #checkpoint_embedding_model = retrieve_hyparams["caption_model"].get("model_path", "sentence-transformers/all-mpnet-base-v2")
        
        # model_path = "liuhaotian/llava-v1.5-7b"
        # model_base = None
        # load_8bit = True

        model_path = retrieve_hyparams["caption_model"].get("model_path", "liuhaotian/llava-v1.5-7b")
        model_base = retrieve_hyparams["caption_model"].get("model_base", None)
        load_8bit = retrieve_hyparams["caption_model"].get("load_8bit", True)
        caption_model = LLAVAModel(model_path, model_base, load_8bit=load_8bit)
    else:
        caption_model = None

    # from models.llava_model_hf import LLAVAModel
    # caption_model = LLAVAModel()
    #caption_model = ''
    ###########################################################################
    # TODO log hyparams to out file
    log_hyparams_with_path(interactive_prompt)


    # def footprint(base_model, caption_model):
    #     print("#" * 25 + "Footprint" + "#" * 25)
    #     # Footprints
    #     # Base Model
    #     print("Base Model Footprint")
    #     try:
    #         base_model_footprint = base_model.model.get_memory_footprint() / 1024 / 1024 / 1024
    #         print(base_model_footprint)
    #     except:
    #         print("No base model footprint")
            
    #     print("Caption Model Footprint")
    #     try:
    #         caption_model_footprint = caption_model.model.get_memory_footprint() / 1024 / 1024 / 1024
    #         print(caption_model_footprint)
    #     except:
    #         print("No caption_model footprint")

    #     try:
    #         print("Allocated Memory")
    #         import torch
    #         print(torch.cuda.memory_allocated())
    #         print(torch.cuda.max_memory_allocated())
    #     except:
    #         pass
        
    #     print("#" * 25 + "" + "#" * 25)
    # footprint(base_model, caption_model)

    if len(dataset) < 3:
        print("dataset: ", dataset)

    
    all_actions = []
    for key in tqdm(dataset, total=len(dataset),
                                     desc='WebQA', disable=disable_tqdm):
        
        example = dataset[key]
        print("#" * 25 + "Case Number: "+ str(example["Guid"]) + "#" * 25)
        ###########################################################################

        example["embedding_model"] = embedding_model
        example["caption_model"] = caption_model

        ###########################################################################


        algo_output = reasoner(example)

        ####  retrieve output
        predicted_sources = [
            source
            for winning_state in algo_output.terminal_state if winning_state.state_type == "RETRIEVE"
            for source in winning_state.retrieved_sources
        ]
        dataset[key]['sources'] = predicted_sources
        #x.state[-1].state_type == "RETRIEVE"
        ####

        answer = algo_output.terminal_state[-1].main_answer
        A = example["A"]

        all_actions.extend(
            winning_state.state_type for winning_state in algo_output.terminal_state
        )
        
        # print("#" * 25 + "Terminal State Start" + "#" * 25)
        # print(algo_output.terminal_state)
        # print("#" * 25 + "Terminal State End" + "#" * 25)

        # print("#" * 25 + "Trace State Start" + "#" * 25)
        # print(algo_output.trace)
        # print("#" * 25 + "trace State End" + "#" * 25)

        print(f'Prediction: {answer}')
        print(f'Answer: {A}')
        log_str = f'Guid: {example["Guid"]} Prediction: {answer} Ground-truth: {A}'


        dataset[key]['answer'] = answer[0]
        dataset[key].pop('path', None)

        dataset[key].pop('instruct_blip', None)
        dataset[key].pop('clip_processor', None)
        dataset[key].pop('clip_model', None)
        dataset[key].pop('embedding_model', None)
        dataset[key].pop('caption_model', None)

        with open(os.path.join(log_dir, f'webqa.json'), 'w') as json_file:
            json.dump(dataset, json_file, indent=4)    

        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
              
            with open(os.path.join(log_dir, 'algo_output', f'{example["Guid"]}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)
            if isinstance(search_algo, MCTS):
                
                with open(os.path.join(log_dir, 'algo_output', f'{example["Guid"]}.json'), 'w') as f: 
                    # noinspection PyTypeChecker
                    print(TreeLog.from_mcts_results(algo_output, node_data_factory=node_visualizer), file=f)

    #with open(os.path.join(log_dir, f'{time_prefix}-webqa.json'), 'w') as json_file:
    with open(os.path.join(log_dir, f'webqa.json'), 'w') as json_file:
        json.dump(dataset, json_file, indent=4)


    from collections import Counter, defaultdict

    count_all_actions = Counter(all_actions)

    avg_all_actions = {
        count_all_actions[key] / len(dataset) for key in count_all_actions
    }

    wandb.summary.update({
        "Action Count": count_all_actions,
        "Action Average": avg_all_actions,
    })

    # function = config.action_selection
    # print("Average time:", function.avg_time())

    # TODO local eval
    #if split == "test":
    if interactive_prompt["general"]["eval-test"]:
        from eval import evaluate
        output = evaluate(dataset, "test")['submission_result']

        wandb.summary.update(output)

    if interactive_prompt["general"]["eval-val"]:

        from pathlib import Path
        with open(Path(path_to_webqa), 'r') as file:
            webqa_data = json.load(file)

        url = "http://127.0.0.1:8000/evaluate"

        data_to_send = {
            "data": dataset
        }

        # Convert your data to JSON
        data_json = json.dumps(data_to_send)
        import requests
        # Send POST request
        response = requests.post(url, data=data_json)

        #print(response.json())
        payload = response.json()
        metrics = payload["metrics"]
        #wandb.summary.update({"Retrieval":f1_score, **metrics})
        wandb.summary.update({**metrics})

        updated_dataset = payload["updated_data"]
        with open(os.path.join(log_dir, f'webqa-evaluated.json'), 'w') as json_file:
            json.dump(updated_dataset, json_file, indent=4)


        



if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama'] = 'llama-2',
             llama_ckpts: str = llama_ckpts,
             llama_2_ckpts: str = llama_2_ckpts,
             llama_size: str = '13B',
             llama_cpp_path: str = None,
             llama_cpp_n_batch: int = 512,
             hf_path: str = 'meta-llama/Llama-2-13b-hf',
             hf_peft_path: Optional[str] = None,
             hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
             hf_load_awq_path: Optional[str] = None,
             exllama_model_dir: str = 'WizardMath-13B-V1.0-GPTQ',
             exllama_lora_dir: Optional[str] = None,
             exllama_mem_map: Optional[str] = None,
             batch_size: int = 1,
             interactive_prompt: str = 'examples/rap_gsm8k/prompts/interactive_examples.json',
             useful_prompt: str = 'examples/rap_gsm8k/prompts/useful_examples.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             seed : int = None,
             **kwargs):
        with open(interactive_prompt) as f:
            interactive_prompt = json.load(f)

            

        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        if base_lm in ['llama', 'llama2']:
            import torch
            import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

        # import torch
        # import torch.backends.cudnn
        # np.random.seed(0)
        # random.seed(0)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        # torch.backends.cudnn.deterministic = True

        # TODO
        print("base_lm: ", base_lm)
        print("exllama_model_dir: ", exllama_model_dir)
        print("hf_path: ", hf_path)

        #     config = {
        #     "LLM"
        #     "Action.RETRIEVE" : retrieve_hyparams,
        #     "ActionSelection" : action_selection,
        # }
        wandb.config.update({
            "CLI": locals()
        })
        # wandb.config.update({
        #     "Configs & Prompts": interactive_prompt
        # })
        if seed is not None:
            
            import torch
            import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True


        if base_lm == 'llama':
            from reasoners.lm import LlamaModel
            base_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama.cpp':
            from reasoners.lm import LlamaCppModel
            base_model = LlamaCppModel(llama_cpp_path, n_batch=llama_cpp_n_batch)
        elif base_lm == 'llama-2':
            from reasoners.lm import Llama2Model
            base_model = Llama2Model(llama_2_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'hf':
            from reasoners.lm import HFModel
            # wandb.config.update({
            #     "LLM": {
            #         "hf_path": hf_path,
            #         "hf_peft_path": hf_peft_path,
            #         "hf_quantized": hf_quantized,
            #     }
            # })
            base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=512,
                                 peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'exllama':
            from reasoners.lm import ExLlamaModel
            base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir, mem_map=exllama_mem_map,
                                      max_batch_size=batch_size, max_new_tokens=200, max_seq_length=2048)
        elif base_lm == "openai":
            from reasoners.lm import GPTCompletionModel
            base_model = GPTCompletionModel(model=exllama_model_dir, temperature=0.7, max_tokens=1000)
        elif base_lm == "fake-llm":
            from reasoners.lm import FakeLLM
            base_model = FakeLLM()
        else:
            assert False, f'cannot resolve {base_lm=}'

        rap_webqa(base_model=base_model,
                  interactive_prompt=interactive_prompt,
                  useful_prompt=useful_prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)

    fire.Fire(main)
