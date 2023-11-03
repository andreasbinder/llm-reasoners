import pickle
from typing import Type, Callable, Optional, Literal

import numpy as np
from datasets import load_dataset
from reasoners.visualization import TreeLog
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation

from world_model import GSM8kWorldModel, GSM8kState, GSM8kAction
from search_config import GSM8kConfig
import utils


def node_visualizer(x: MCTSNode[GSM8kState, GSM8kAction]):
    if not x.state:
        return {}
    # return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
    # match states
    if x.state[-1].state_type == "ANSWER":
        return {"question": x.state[-1].main_question, "answer": x.state[-1].main_answer}
    elif x.state[-1].state_type == "RETRIEVE":
        #, "retrieved_snippets": x.state[-1].retrieved_snippets
        return {
            "context": x.state[-1].context,
            "snippet_id": [snippet_id.split("_")[-1] for snippet_id in x.state[-1].retrieved_sources],
            "flags": x.state[-1].flags
        } 
    elif x.state[-1].state_type == "INVALID":
        return {"content": "INVALID"} 


def rap_gsm8k(base_model: LanguageModel,
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
              cum_reward: Callable[[list[float]], float] = np.mean,
              calc_q: Callable[[list[float]], float] = max,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              output_trace_in_each_iter: bool = True,
              aggregate: bool = True,
              path_to_webqa: str = None, # TODO
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
    world_model = GSM8kWorldModel(base_model=base_model, prompt=interactive_prompt,
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

    
 
    HF_memory_footprint = base_model.model.model.get_memory_footprint() if hasattr(base_model.model, 'get_memory_footprint') else None
    print("HF_memory_footprint: ", HF_memory_footprint)
    
    #path_to_webqa = '/home/stud/abinder/Multimodal-LLMs-for-webscale-Questions-Answering/data/n_samples_50_split_val_solution_txt_seed_42_1691423190.7960498_samples.json'
    def load_webqa_dataset(path_to_webqa, resume):
        from pathlib import Path
        data = json.loads(Path(path_to_webqa).read_text())
        if isinstance(resume, str):
            start_str, end_str = resume.strip('[]').split(',')
            start, end = int(start_str), int(end_str)
            selected_indices = list(range(start, end+1))
        if isinstance(resume, int):
            selected_indices = [resume]
        if isinstance(resume, list):
            selected_indices = list(range(resume[0], resume[1]+1))
        # - guid
        # - sources
        # - answer
        # - ground-truth
        # - question
        # - Qcate
        
        selected_data = [
            {
                "index": idx, 
                "question": data[idx]["Q"], 
                "Guid": data[idx]["Guid"], 
                "ground-truth": data[idx]["A"],
                "Qcate": data[idx]["Qcate"],
                "path": path_to_webqa
            } 
            for idx in selected_indices if idx < len(data)
        ]
        return selected_data

    dataset = load_webqa_dataset(path_to_webqa, resume)

    webqa_results = {}

    if len(dataset) < 3:
        print("dataset: ", dataset)


    correct_count = 0
    # for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
    #                                  desc='WebQA', disable=disable_tqdm)):
    for i, example in enumerate(tqdm(dataset, total=len(dataset),
                                     desc='WebQA', disable=disable_tqdm)):
        print("#" * 25 + "Case Number: "+ str(i) + "#" * 25)
        
        # algo_output = reasoner(example["question"])
        algo_output = reasoner(example)

        # TODO only necessary for evaluation -> do our own
        # TODO what does the aggregator do?
        # if aggregate:
        #     output = aggregator(algo_output.tree_state)
        # elif algo_output.terminal_state is None:
        #     output = None
        # else:
        #     output = utils.retrieve_answer(algo_output.terminal_state)
        # answer = utils.retrieve_answer_from_dataset(example["answer"])
        # correct = utils.judge_answer(output, answer)

        # correct_count += correct
        # accuracy = correct_count / (i + 1)
        # log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=} ; {accuracy=:.3f} ({correct_count}/{i + 1})'
        prediction = algo_output.terminal_state[-1].main_answer
        print(f'Prediction: {prediction}')
        print(f'Answer: {example["ground-truth"]}')
        log_str = f'Case {i}'
        webqa_results.update({
            example["Guid"]: {
                # for manual inspection
                "Guid": example["Guid"],   
                "Qcate": example["Qcate"],
                "Q": example["question"], 
                "A": example["ground-truth"],
                "Keywords_A" : "TBD",
                # "Output_conf" : [1], # not really needed
                # needed for online evaluation
                "answer": prediction[0], # TODO supposed to not be list
                "sources": [], # TODO
            }
        })

        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
              
            # with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
            with open(os.path.join(log_dir, 'algo_output', f'{time_prefix}-{example["Guid"]}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)
            if isinstance(search_algo, MCTS):
                # with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.json'), 'w') as f:
                with open(os.path.join(log_dir, 'algo_output', f'{time_prefix}-{example["Guid"]}.json'), 'w') as f: 
                    # noinspection PyTypeChecker
                    print(TreeLog.from_mcts_results(algo_output, node_data_factory=node_visualizer), file=f)

    with open(os.path.join(log_dir, f'{time_prefix}-webqa.json'), 'w') as json_file:
        json.dump(webqa_results, json_file, indent=4)

    # create offline version 
    # webqa_results_offline = {}
    # for key in webqa_results:
    #     webqa_results_offline.update({
    #         key: {
    #             "Guid": key,  
    #             "Qcate" : webqa_results[key]["Qcate"], 
    #             "Q" : webqa_results[key]["question"],
    #             "A" : webqa_results[key]["ground-truth"], 
    #             "Keywords_A" : "TBD",
    #             # "Output_conf" : [1], # not really needed
    #             "Output" : webqa_results[key]["answer"], # is already a list                
    #         }
    #     }
    # )
    # with open(os.path.join(log_dir, f'{time_prefix}-webqa-offline.json'), 'w') as json_file:
    #     json.dump(webqa_results_offline, json_file, indent=4)


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

        # TODO
        print("base_lm: ", base_lm)
        print("exllama_model_dir: ", exllama_model_dir)
        print("hf_path: ", hf_path)

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
            base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=512,
                                 peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'exllama':
            from reasoners.lm import ExLlamaModel
            base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir, mem_map=exllama_mem_map,
                                      max_batch_size=batch_size, max_new_tokens=200, max_seq_length=2048)
        elif base_lm == "openai":
            from reasoners.lm import GPTCompletionModel
            base_model = GPTCompletionModel(model=exllama_model_dir, temperature=0.7, max_tokens=1000)
        else:
            assert False, f'cannot resolve {base_lm=}'

        rap_gsm8k(base_model=base_model,
                  interactive_prompt=interactive_prompt,
                  useful_prompt=useful_prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)

    fire.Fire(main)
