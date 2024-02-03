import io
import re
from typing import TypedDict, Optional

import numpy as np

from world_model import GSM8kState, WebQAAction, GSM8kPrompt, DecomposeResult, RetrievalResult
from reasoners import SearchConfig, LanguageModel

import utils

import wandb

import time

def time_decorator(func):
    total_time = 0
    call_count = 0
    results = []

    def wrapper(*args, **kwargs):
        nonlocal total_time, call_count
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        call_count += 1
        total_time += (end_time - start_time)
        avg_time = total_time / call_count

        results.append(result)

        print(f"{func.__name__} executed in {(end_time - start_time):.4f} seconds, running average: {avg_time:.4f} seconds")

        return result

    wrapper.results = results
    wrapper.avg_time = lambda: total_time / call_count if call_count else float('inf')

    return wrapper


class GSM8kUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str


def iterative_decompose(config, example, base_model, temperature):
            
    decompose_prompt = config["evaluation"]["prompts"]["iterative_decompose"].format(
        Q=example["Q"],
    )
    n = config["general"]["iterative_decompose_max_steps"]
    
    print(decompose_prompt)
    print("Start Decompose")
    subs = []
    for i in range(1, n+1):
        decompose_prompt += "Question 3.{i}:"
        sub = base_model.generate([decompose_prompt],
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                min_new_tokens=3,
                                                eos_token_id='\n').text[0]
        print(f"Sub {i}: {sub.strip()}")
        subs.append(sub.strip())
        decompose_prompt += sub.strip()


    if config["general"]["use_main_question"]:
        subs = [example["Q"]] + subs
        # 
    print("End Decompose")
    return subs

class WebQAConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 batch_size=1,
                 temperature=0.8,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = ''
        self.prompt: GSM8kPrompt = prompt
        
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        self.overall_question: Optional[str] = None

        self.action_selection_hyparams = self.prompt["action_selection"]["hyparams"]
        self.max_attempts = self.action_selection_hyparams.get("max_attempts", 3)
        self.generation_cutoff = self.action_selection_hyparams.get("generation_cutoff", 3)

        self.general_hyparams = self.prompt["general"]["hyparams"]
        self.min_new_tokens = self.general_hyparams.get("min_new_tokens", 3)

        self.subs = []

    def update_example(self, example: str) -> None:
        super().update_example(example)
        # TODO
        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example)[1]
            self.overall_question = self.example

        if self.prompt['general']['use_iterative_decompose']:
            self.subs = iterative_decompose(self.prompt, self.example, self.base_model, self.temperature)

    @time_decorator
    def action_selection(self, state: GSM8kState) -> DecomposeResult:

        question = self.example["Q"]

        available_actions = self.prompt["action_selection"]["available_actions"]

        model_input = utils.action_selection_prompt(self.prompt, question, state, available_actions)


        

        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0.0001 if at_depth_limit else self.temperature #TODO
        selected_actions = []

        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            
            for i in range(self.max_attempts):
                
                #logits = self.base_model.get_next_token_logits(model_input, ["ANSWER", "Answer", "RETRIEVE", "Retrieve"])[0]

                selected_action = self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                max_new_tokens=self.generation_cutoff,
                                                min_new_tokens=self.min_new_tokens,
                                                eos_token_id='\n').text[0]

                print(f"Complete Selected Action: '{selected_action}'")                                                
                keyword = utils.find_first_appearance(selected_action, available_actions) 
                if keyword is None:
                    temperature = self.temperature
                    print(f"Output '{selected_action}' not a valid action. Retrying...")
                    continue  # This skips the rest of the code in the loop and starts the next iteration

                # Process the valid output
                print(f"Valid action received: {keyword}")
                #print(f"Logits: {logits}")
                # ... your code to handle the valid output ...
                selected_actions += [keyword]
                break  # Break the loop if a valid action is found
            else:
                print("No valid action was found after maximum attempts.")
                selected_actions += ["INVALID"]

        #keywords = ["HYPOTHESIS"] * 4
        return selected_actions

        
    
    # @time_decorator
    # def action_selection(self, state: GSM8kState) -> DecomposeResult:


    #     question = self.example["Q"]

    #     available_actions = self.prompt["action_selection"]["available_actions"]

    #     model_input = utils.action_selection_prompt(self.prompt, question, state, available_actions)


        

    #     at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
    #     n_actions = 1 if at_depth_limit else self.n_actions
    #     temperature = 0.0001 if at_depth_limit else self.temperature #TODO
    #     selected_actions = []

    #     n_samples = n_actions #min(n_actions, self.batch_size)
                
    #     logits = self.base_model.get_next_token_logits(model_input, ["ANSWER", "Answer", "RETRIEVE", "Retrieve"])[0]
    #     print(f"Logits: {logits}")

    #     selected_actions = self.base_model.generate([model_input] * n_samples,
    #                                     hide_input=True,
    #                                     do_sample=True,
    #                                     temperature=temperature,
    #                                     max_new_tokens=self.generation_cutoff,
    #                                     min_new_tokens=self.min_new_tokens,
    #                                     eos_token_id='\n').text
    #     keywords = []
    #     for selected_action in selected_actions:
                
    #         print(f"Complete Selected Action: '{selected_action}'")                                                
    #         keyword = utils.find_first_appearance(selected_action, available_actions) 
    #         if keyword is None:
    #             keywords += ["INVALID"]
    #             print(f"Output '{selected_action}' not a valid action. Retrying...")
    #         else:

    #             # Process the valid output
    #             print(f"Valid action received: {keyword}")
                
    #             # ... your code to handle the valid output ...
    #             keywords += [keyword]

    #     return keywords


    def get_actions(self, state: GSM8kState, ) -> list[WebQAAction]:

        question = self.example["Q"]

        # if len(state) == 0 and self.prompt["general"]["answer_limit"]:
        #     available_actions = ['RETRIEVE']
        # else:
        #     available_actions = self.prompt["action_selection"]["available_actions"]

        # if state == []:
        #     available_actions = ['RETRIEVE']
        # elif len(state) == 1: 
        #     available_actions = ['REFINE']
        # else:
        #     available_actions = self.prompt["action_selection"]["available_actions"]

        if state == []:
            available_actions = ['RETRIEVE', 'ANSWER']
        else:
            available_actions = self.prompt["action_selection"]["available_actions"]

        #available_actions = self.prompt["action_selection"]["available_actions"]

        model_input = utils.action_selection_prompt(self.prompt, question, state, available_actions)


        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0.0001 if at_depth_limit else self.temperature #TODO
        selected_actions = []

        # TODO improvement to avoid unnecessary generation
        if at_depth_limit:

            # actions = ['ANSWER' + ': ' + question for _ in actions]
            actions = [('ANSWER', question) for _ in range(n_actions)]
            # set does not guarantee order, but dict does guarantee
            # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
            actions = list(dict.fromkeys(actions))
            return actions

        
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            # selected_actions += self.base_model.generate([model_input] * n_samples,
            #                                     hide_input=True,
            #                                     do_sample=True,
            #                                     temperature=temperature,
            #                                     eos_token_id='\n').text
            for i in range(self.max_attempts):
                
                selected_action = self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                max_new_tokens=self.generation_cutoff,
                                                min_new_tokens=self.min_new_tokens,
                                                eos_token_id='\n').text[0]

                print(f"Complete Selected Action: '{selected_action}'")                                                
                keyword = utils.find_first_appearance(selected_action, available_actions) 
                if keyword is None:
                    temperature = self.temperature
                    print(f"Output '{selected_action}' not a valid action. Retrying...")
                    continue  # This skips the rest of the code in the loop and starts the next iteration

                # Process the valid output
                print(f"Valid action received: {keyword}")
                # ... your code to handle the valid output ...
                selected_actions += [keyword]
                break  # Break the loop if a valid action is found
            else:
                print("No valid action was found after maximum attempts.")
                selected_actions += ["INVALID"]

        # keywords = ["HYPOTHESIS"] * 4
        keywords = selected_actions

        # keywords = self.action_selection(state)

        
        print("#" * 25 + "Action Selection" + "#" * 25)
        for item1, item2 in zip([model_input] * n_actions, keywords):            
            print(item1, item2, sep='\n')
            print("#" * 25)
        #keywords = ["RETRIEVE"]

        # print("keywords")
        # print(keywords)
        
        prompts_per_keyword = [utils.action_prompt(self.prompt, question, state, keyword) if keyword != 'INVALID' else 'INVALID' for keyword in keywords]
        
        # print("#" * 25 + "Action Input" + "#" * 25)
        # print(prompts_per_keyword)

        actions = []
        for keyword, prompt in zip(keywords, prompts_per_keyword):
            if keyword == 'ANSWER':
                #actions += ['ANSWER' + ': ' + question]
                actions += [('ANSWER', question)]
            
            elif keyword == 'INVALID':
                #actions += ['INVALID']
                actions += [('INVALID', 'No valid Keyword generated')]
            elif keyword == 'REFINE':
                #actions += ['REFINE' + ': ' + question]
                actions += [('REFINE', '')]
            else:
                model_output = self.base_model.generate([prompt],
                                                    hide_input=True,
                                                    do_sample=True,
                                                    temperature=temperature,
                                                    min_new_tokens=self.min_new_tokens,
                                                    eos_token_id='\n').text
                
                
                # actions += [keyword + ': ' + model_output[0]]
                
                #actions += [(keyword, model_output[0])]
                actions += [(keyword, model_output[0].strip())]
                
            # wandb.log({
            #     str(len(state)): {
            #         "Action Input": prompt,
            #         "Action Output": actions[-1][1]
            #     }
            # })

        # print("#" * 25 + "Action Output" + "#" * 25)
        # print(actions)

        
        print("#" * 25 + "Action Enrichment" + "#" * 25)
        for item1, item2 in zip(prompts_per_keyword, actions):

            print(item1, item2, sep='\n')
            print("#" * 25)

        # actions = [action.strip() for action in actions]
        if at_depth_limit:

            # actions = ['ANSWER' + ': ' + question for _ in actions]
            actions = [('ANSWER', question) for _ in actions]
        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        actions = list(dict.fromkeys(actions))
        return actions

    def fast_reward(self, state: GSM8kState, action: WebQAAction) -> tuple[float, dict]:
        
        
        question = self.example["Q"]
        keyword, details = action

        # TODO make sure if works as intended
        if keyword == 'INVALID':
            return 0, {'r_useful': 0}

        model_input = utils.evaluation_prompt(self.prompt, question , state, action)

        print("#" * 25 + "Evaluation Input" + "#" * 25)
        print(model_input)

        # logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        # probs = np.exp(logits) / np.sum(np.exp(logits))
        # useful_prob = probs[0]
        if self.prompt["general"]["use_ensemble"]:
            useful_prob = self.ensemble_reward(model_input)
        elif self.prompt["general"]["use_iterative_decompose"]:
            useful_prob = self.iterative_decompose_reward(state, action)
        else:
            useful_prob = self.single_reward(model_input)
        print("#" * 25 + "Evaluation Output" + "#" * 25)
        print(useful_prob)

        # if len(state) == 0 and keyword == 'ANSWER' and self.prompt["general"]["answer_reward"]:
        #     useful_prob *= 0.8

        fast_reward, _ = self.calculate_reward(useful_prob)
        return fast_reward, {'r_useful': useful_prob}

    def single_reward(self, model_input: str):
        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        return useful_prob
    
    def iterative_decompose_reward(self, state, action):
        avg_useful_prob = 0
        print("#" * 25 + "Iterative Reward Start" + "#" * 25)
        for sub in self.subs:
            
            model_input = utils.evaluation_prompt(self.prompt, sub , state, action)
            logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            useful_prob = probs[0]
            print(useful_prob)

            avg_useful_prob += useful_prob
        print("#" * 25 + "Iterative Reward End" + "#" * 25)
        return avg_useful_prob / self.prompt["general"]["ensemble_size"]

    def ensemble_reward(self, model_input: str):
        avg_useful_prob = 0
        print("#" * 25 + "Ensemble Reward Start" + "#" * 25)
        for _ in range(self.prompt["general"]["ensemble_size"]):
            logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            useful_prob = probs[0]
            print(useful_prob)

            avg_useful_prob += useful_prob
        print("#" * 25 + "Ensemble Reward End" + "#" * 25)
        # logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        # probs = np.exp(logits) / np.sum(np.exp(logits))
        # useful_prob = probs[0]
        return avg_useful_prob / self.prompt["general"]["ensemble_size"]

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}

    def reward(self, state: GSM8kState, action: WebQAAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)


