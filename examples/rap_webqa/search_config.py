import io
import re
from typing import TypedDict, Optional

import numpy as np

from world_model import GSM8kState, GSM8kAction, GSM8kPrompt, DecomposeResult, RetrievalResult
from reasoners import SearchConfig, LanguageModel

import utils

class GSM8kUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str


class GSM8kConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 useful_prompt: dict,
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
        self.useful_prompt: GSM8kUsefulPrompt = useful_prompt
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

    def update_example(self, example: str) -> None:
        super().update_example(example)
        # TODO
        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example)[1]
            self.overall_question = self.example

    def get_actions(self, state: GSM8kState, ) -> list[GSM8kAction]:

        model_input = utils.action_selection_prompt(self.prompt, self.example, state)

        #model_input = action_selection()
        print("#" * 25 + "Action Selection Input" + "#" * 25)
        print(model_input)

        available_actions = list(self.prompt["actions"].keys())

        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0 if at_depth_limit else self.temperature
        selected_actions = []

        max_attempts = 5
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            # selected_actions += self.base_model.generate([model_input] * n_samples,
            #                                     hide_input=True,
            #                                     do_sample=True,
            #                                     temperature=temperature,
            #                                     eos_token_id='\n').text
            for i in range(max_attempts):
                
                selected_action = self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                eos_token_id='\n').text[0]
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
        # print("#" * 25 + "Action Selection Output" + "#" * 25)
        # print(outputs)

        # #first_outputs = [find_first_appearance(output) for output in outputs]
        # keywords = [utils.find_first_appearance(selected_action, self.prompt["actions"].keys()) for selected_action in selected_actions]
        keywords = selected_actions
        

        # keywords = ['DECOMPOSE', 'DECOMPOSE', 'DECOMPOSE', 'DECOMPOSE']
        # keywords = ['RETRIEVE', 'RETRIEVE', 'RETRIEVE', 'RETRIEVE']
        # keywords = ['RETRIEVE', 'RETRIEVE', 'DECOMPOSE', 'DECOMPOSE']
        # result = ['R1', 'R2', 'D11','D12', 'D21','D22']
        print("keywords")
        print(keywords)
        
        prompts_per_keyword = [utils.action_prompt(self.prompt, self.example, state, keyword) for keyword in keywords]
        #prompts_per_keyword = ["Generate a textual query for finding the university that started offering courses in the community with ZIP code 29707 in August 2018.\n"] * 4
        #prompts_per_keyword = ["Generate a subquestion related to the following question: 'In August 2018, what university began offering courses in the community with ZIP code 29707?\n"] * 4 
        #prompts_per_keyword = ["Generate a subquestion that gives a partial answer to the following question: Did Emperor Heraclius fight against the Fifth Dynasty of ancient Egypt?\n"] * 4 
        #################

        #################
        print("#" * 25 + "Action Input" + "#" * 25)
        print(prompts_per_keyword)

        actions = []
        for keyword, prompt in zip(keywords, prompts_per_keyword):
            if keyword == 'ANSWER':
                actions += ['ANSWER' + ': ' + self.example]
            else:
                model_output = self.base_model.generate([prompt],
                                                    hide_input=True,
                                                    do_sample=True,
                                                    temperature=temperature,
                                                    eos_token_id='\n').text
                
                actions += [keyword + ': ' + model_output[0]]
                
        print("#" * 25 + "Action Output" + "#" * 25)
        print(actions)

        # import sys
        # sys.exit()

        actions = [action.strip() for action in actions]
        if at_depth_limit:
            
            #outputs = ['ANSWER' for output in outputs]
            actions = ['ANSWER' + ': ' + self.example for _ in actions]
        # TODO understand use

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        actions = list(dict.fromkeys(actions))
        return actions

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        
        model_input = utils.evaluation_prompt(self.prompt, self.useful_prompt, self.example, state, action)

        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        fast_reward, _ = self.calculate_reward(useful_prob)
        return fast_reward, {'r_useful': useful_prob}

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}

    def reward(self, state: GSM8kState, action: GSM8kAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)
