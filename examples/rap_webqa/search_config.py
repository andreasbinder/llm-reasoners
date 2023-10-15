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

        # state_list = [
        #     DecomposeResult("What is Nevezis exactly?", "The Nevezis is the sixth longest river in Lithuania", 0.9),
        #     RetrievalResult("Give me the length of the A1 Kaunas-Klaipeda highway.", "311,4 km.", []),
        # ]
        # state_list = [
        #     DecomposeResult("Who are the best tree travelers in the animal kingdom?", "Gibbons are the best tree travelers in the animal kingdom.", 0.9),
        #     RetrievalResult("Give me the direction the Salween River in Myanmar flows.", "It flows from North to South.", []),
        # ]
        # state_list = [
        #     #DecomposeResult("Who are the best tree travelers in the animal kingdom?", "Gibbons are the best tree travelers in the animal kingdom.", 0.9),
        #     RetrievalResult("Is it very cloudy on the \"Summer, Lake Ontario\" painting by Jasper Francis Cropsey?", "It is not very cloudy.", []),
        # ]
        state_list = [
            #DecomposeResult("Who are the best tree travelers in the animal kingdom?", "Gibbons are the best tree travelers in the animal kingdom.", 0.9),
            RetrievalResult("List the universities with the ZIP code 29707?", "The University of South Carolina Lancaster and University of Manhattan", []),
        ]
        state_list = []
        model_input = utils.action_selection(self.prompt, self.example, state_list)

        #model_input = action_selection()
        print("#" * 25 + "Action Selection Input" + "#" * 25)
        print(model_input)

        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0 if at_depth_limit else self.temperature
        outputs = []
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            outputs += self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                eos_token_id='\n').text


        print("#" * 25 + "Action Selection Output" + "#" * 25)
        print(outputs)

        

        def find_first_appearance(text):
            keywords = list(self.prompt["actions"].keys())
            
            for keyword in keywords:
                if keyword in text:
                    return keyword
            
            return None  # None of the keywords found

        first_outputs = [find_first_appearance(output) for output in outputs]

        
        #first_outputs = ['DECOMPOSE', 'DECOMPOSE', 'DECOMPOSE', 'DECOMPOSE']
        first_outputs = ['RETRIEVE', 'RETRIEVE', 'RETRIEVE', 'RETRIEVE']

        print("first_outputs")
        print(first_outputs)
        
        actions = [utils.execute_action(self.prompt, self.example, state_list, output) for output in first_outputs]

        print("#" * 25 + "Action Input" + "#" * 25)
        print(actions)


        outputs = []
        for act in actions:
            if act != 'ANSWER':
                outputs += self.base_model.generate([act],
                                                    hide_input=True,
                                                    do_sample=True,
                                                    temperature=temperature,
                                                    eos_token_id='\n').text
                
            else:
                outputs += ['ANSWER']

        # 
        print("#" * 25 + "Action Output" + "#" * 25)
        print(outputs)

        # import sys
        # sys.exit()
        

        outputs = [output.strip() for output in outputs]
        # if at_depth_limit:
        #     outputs = [self.prompt["overall_question_prefix"] + ' ' + output for output in outputs]
        # TODO understand use
        # if self.force_overall_question_on_overall_prompt:
        #     for i, output in enumerate(outputs):
        #         if self.prompt["overall_question_prefix"] in output:
        #             outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question
        # if self.force_overall_prompt_on_overall_question:
        #     for i, output in enumerate(outputs):
        #         if self.overall_question.lower() == output.lower():
        #             outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        outputs = list(dict.fromkeys(outputs))
        return outputs

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, _, _) in enumerate(state):
                f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
            f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.useful_prompt["useful_prefix"])
            model_input = f.getvalue()

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
