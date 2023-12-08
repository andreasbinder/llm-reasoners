import io
import re
from typing import TypedDict, Optional

import numpy as np

from world_model import GSM8kState, WebQAAction, GSM8kPrompt, DecomposeResult, RetrievalResult
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

    def get_actions(self, state: GSM8kState, ) -> list[WebQAAction]:

        question = self.example["Q"]

        model_input = utils.action_selection_prompt(self.prompt, question, state)

        #model_input = action_selection()
        print("#" * 25 + "Action Selection Input" + "#" * 25)
        print(model_input)

        available_actions = list(self.prompt["actions"].keys())

        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0.0001 if at_depth_limit else self.temperature #TODO
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
                                                #eos_token_id='\n'
                                                ).text[0]
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

        keywords = selected_actions
        
        #keywords = ["RETRIEVE"]

        print("keywords")
        print(keywords)
        
        #prompts_per_keyword = [utils.action_prompt(self.prompt, question, state, keyword) if keyword != 'INVALID' else 'INVALID' for keyword in keywords]prompts_per_keyword = [utils.action_prompt(self.prompt, question, state, keyword) if keyword != 'INVALID' else 'INVALID' for keyword in keywords]
        
        # TODO hardcode prompts for now
        retrieve_prompt = f"""\
        [INST] Create a textual query to answer an overall question. [/INST] \
            Sure Thing! \
        [INST] Are both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida the same color? [/INST] \
            Compare the exterior colors of the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida. \
        [INST] What water-related object is sitting in front of the Torre del Reloj? [/INST] \
            Identify the water-related object located in front of the Torre del Reloj. \
        [INST] {question} [/INST] \ 
        """

        def get_history(state_list, action):
            # if action == "RETRIEVE":
            #     return state.retrieved_snippets
            if state_list != []:
                snippets = [snippet for state in state_list for snippet in state.retrieved_snippets]
                out = ""
                out += "This context you can use to answer the question: "
                for idx, snippet in enumerate(snippets):
                    out += f"{idx}) {snippet}" + " "
                return out

        answer_prompt = f"""\
        [INST] Provide a direct answer to the question without forwarding a task. \
            The overall question is: {question} \
        [INST] Are both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida the same color? [/INST] \
            Compare the exterior colors of the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida. \
        [INST] What water-related object is sitting in front of the Torre del Reloj? [/INST] \
            Identify the water-related object located in front of the Torre del Reloj. \
        [INST] {get_history(state, "RETRIEVEs")}  \ 
            Give a short and concise answer. [/INST]
        """

        # prompts_per_keyword = [retrieve_prompt if keyword == "RETRIEVE" else answer_prompt for keyword in keywords]
        # prompts_per_keyword = ["INVALID" if keyword == "RETRIEVE" else expression2 if condition2 else expression3 for keyword in keywords]
        prompts_per_keyword = []
        for keyword in keywords:
            if keyword == "RETRIEVE":
                prompts_per_keyword.append(retrieve_prompt)
            elif keyword == "ANSWER":
                prompts_per_keyword.append(answer_prompt)
            else:
                prompts_per_keyword.append("INVALID")
        # The overall question is: "What type of stem do the flowers of the Barringtonia asiatica grow from?"

        # 0) The image features a plant with green leaves and a white flower. The plant is surrounded by several bananas, some of which are green and unripe. The bananas are in various positions, with some hanging from the plant and others placed on the ground. The overall scene showcases the growth and development of the plant and its fruit.
        # 1) The image shows a field of green plants, some of which appear to be tobacco plants. The plants are growing in a dirt field, and their leaves are large and green.
        # 2) The image features a tree filled with yellow fruits, possibly lemons or oranges, hanging from its branches. The tree is surrounded by a blue sky, creating a vibrant and lively scene.
        # 3) On the lagoon side, it may also contain Lepturus repens, Triumfetta procumbens and Cyperus ligularis. Large pockets of Barringtonia asiatica are also on the eastern edge of the lagoon.
        # The final answer is:

        print("#" * 25 + "Action Input" + "#" * 25)
        print(prompts_per_keyword)

        actions = []
        for keyword, prompt in zip(keywords, prompts_per_keyword):
            if keyword == 'ANSWER':
                #actions += ['ANSWER' + ': ' + question]
                actions += [('ANSWER', question)]
            
            elif keyword == 'INVALID':
                #actions += ['INVALID']
                actions += [('INVALID', 'No valid Keyword generated')]
            else:
                model_output = self.base_model.generate([prompt],
                                                    hide_input=True,
                                                    do_sample=True,
                                                    temperature=temperature,
                                                    eos_token_id='\n').text
                
                # actions += [keyword + ': ' + model_output[0]]
                actions += [(keyword, model_output[0])]
                
        print("#" * 25 + "Action Output" + "#" * 25)
        print(actions)

        # actions = [action.strip() for action in actions]
        if at_depth_limit:

            # actions = ['ANSWER' + ': ' + question for _ in actions]
            actions = [('ANSWER', question) for _ in actions]
        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        actions = list(dict.fromkeys(actions))
        return actions

    def fast_reward(self, state: GSM8kState, action: WebQAAction) -> tuple[float, dict]:
        
        #model_input = utils.evaluation_prompt(self.prompt, self.useful_prompt, question , state, action)
        question = self.example["Q"]
        keyword, details = action

        # TODO make sure if works as intended
        if keyword == 'INVALID':
            return 0, {'r_useful': 0}

        model_input = utils.evaluation_prompt(self.prompt, self.useful_prompt, question , state, details)

        print("#" * 25 + "Evaluation Input" + "#" * 25)
        print(model_input)

        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        print("#" * 25 + "Evaluation Output" + "#" * 25)
        print(useful_prob)


        fast_reward, _ = self.calculate_reward(useful_prob)
        return fast_reward, {'r_useful': useful_prob}

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
