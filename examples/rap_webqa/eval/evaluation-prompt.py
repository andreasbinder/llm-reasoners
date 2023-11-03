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
        if log_dir is None:
            log_dir = f'logs/webqa_{search_algo.__name__}_eval/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        #os.makedirs(log_dir, exist_ok=resume > 0)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    evaluation_prompt = '''
    // System define Imagine you are a strict marking teacher or grader.  \n
    I will give you a question, a correct answer, and a student answer. \n
    You need to tell me "1" or "0" (where 1 means correct, 0 means incorect). \n
    "1" does not mean the student's answer must exactly match the correct answer. \n
    If they have the same meaning for the given question, then it is also "1". \n
    However, an ambiguous answer is "0" (e.g., correct answer: "1789", student answer: "long long ago"). \n

    Input Question: {question} \n

    Correct Answer: {reference} \n

    Answer: {prediction} \n

    Grade:    
    '''
    
    HF_memory_footprint = base_model.model.model.get_memory_footprint() if hasattr(base_model.model, 'get_memory_footprint') else None
    print("HF_memory_footprint: ", HF_memory_footprint)
    
    #path_to_webqa = '/home/stud/abinder/Multimodal-LLMs-for-webscale-Questions-Answering/data/n_samples_50_split_val_solution_txt_seed_42_1691423190.7960498_samples.json'
    def load_webqa_dataset(path_to_webqa, resume):
        from pathlib import Path
        data = json.loads(Path(path_to_webqa).read_text())
        return data

    dataset = load_webqa_dataset(path_to_webqa, resume)

    # dataset = {key: dataset[key] for i, key in enumerate(dataset) if i < 2}

    webqa_results_scored = {}

    correct_count = 0
    # for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
    #                                  desc='WebQA', disable=disable_tqdm)):

    def extract_grade(llm_output):
        """
        Extracts the grade (0 or 1) from the LLM output.

        :param llm_output: A string output from the LLM.
        :return: An integer 0 or 1, or None if neither is found.
        """
        if "1" in llm_output:
            return 1
        elif "0" in llm_output:
            return 0
        else:
            return -1  # or raise an exception if preferred

    for i, key in enumerate(tqdm(dataset.keys(), total=len(dataset),
                                     desc='WebQA', disable=disable_tqdm)):
        print("#" * 25 + "Case Number: "+ str(i) + "#" * 25)
        

        model_input = evaluation_prompt.format(
            question=dataset[key]["question"],
            reference=dataset[key]["answer"],
            prediction=dataset[key]["prediction"]
        )
        print(model_input)

        # algo_output = reasoner(example["question"])
        judgement = base_model.generate([model_input],
            hide_input=True,
            do_sample=True,
            temperature=temperature,
            eos_token_id='\n'
        ).text[0]

        grade = extract_grade(judgement)

        print(f'Question: {dataset[key]["question"]}')
        print(f'Prediction: {dataset[key]["prediction"]}')
        print(f'Answer: {dataset[key]["answer"]}')
        print(f'Judgement: {judgement}')
        print(f'Grade: {grade}')
        
        log_str = f'Case {i}'
        webqa_results_scored.update({
            key: {
                "prediction": dataset[key]["prediction"],
                "answer": dataset[key]["answer"],
                "question": dataset[key]["question"], 
                "judgement": judgement,
                "score": grade,
            }
        })

        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)

    correct_count = sum([1 for key in webqa_results_scored if webqa_results_scored[key]["score"] == 1])
    wrong_count = sum([1 for key in webqa_results_scored if webqa_results_scored[key]["score"] == 0])
    none_count = sum([1 for key in webqa_results_scored if webqa_results_scored[key]["score"] == -1])

    print("Summary: ")
    print("Total Correct: ", correct_count)
    print("Total Wrong: ", wrong_count)
    print("Total None: ", none_count)

    webqa_results_scored.update({
            "summary": {
                "TotalCorrect": correct_count,
                "Total Wrong": wrong_count,
                "Total None": none_count, 
            }
        })

    with open(os.path.join(log_dir, f'webqa_results_scored.json'), 'w') as json_file:
        json.dump(webqa_results_scored, json_file, indent=4)

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
