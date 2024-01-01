import io
from typing import NamedTuple, TypedDict, Union, Tuple, List
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils

from tools import RetrievalResult, Retrieval
#from tools.retrieval_base import RetrievalResult, CrossRetrieval as Retrieval
#from tools.clip_ib_retrieval import ClipRetrieval as Retrieval

from collections import namedtuple

class DecomposeResult(NamedTuple):
    state_type: str
    sub_question: str
    sub_answer: str
    confidence: float


class HypothesisResult(NamedTuple):
    state_type: str
    proposition: str
    comment: str
    confidence: float

class AnswerResult(NamedTuple):
    state_type: str
    main_question: str
    main_answer: str
    confidence: float

class InvalidResult(NamedTuple):
    state_type: str
    main_answer: str
    


GSM8kState = list[Union[RetrievalResult, DecomposeResult, AnswerResult, InvalidResult]]
# WebQAAction = str
action_keyword = str
action_details = str
WebQAAction = Tuple[action_keyword, action_details]


class GSM8kPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    answer_prefix: str
    overall_question_prefix: str
        

class Answer():
    """
    A class to generate answers for given prompts using a machine learning model.
    
    Attributes:
        base_model (BaseModel): The machine learning model used to generate answers.
        temperature (float): The sampling temperature to use when generating answers.
        question (str): The question for which the answer is generated.
    """

    def __init__(self, base_model, temperature, example) -> None:
        """
        Constructs all the necessary attributes for the AnswerGenerator object.
        
        Parameters:
            base_model (BaseModel): The pre-trained base model for generating answers.
            temperature (float): A coefficient to control the randomness of predictions
                                 by scaling the logits before applying softmax.
            example (dict): A dictionary containing a 'question' key.
        """
        self.base_model = base_model
        self.temperature = temperature
        self.question = example['Q']

    def answer(self, prompt, state: str) -> str:
        """
        Generates an answer for the given prompt.

        Parameters:
            prompt (str): The prompt to which the model should respond.
            state (dict): The current state information to be considered by the model.

        Returns:
            AnswerResult: An object containing the action, question, generated answer, and confidence level.
        """
        
        #model_input = utils.answer_prompt(prompt, self.question, state, "ANSWER")
        
        model_input = utils.action_prompt(prompt, self.question, state, "ANSWER")
        
        print("#" * 25 + "ANSWER Input" + "#" * 25)
        print(model_input)

        answer = self.base_model.generate([model_input],
                                            hide_input=True,
                                            do_sample=True,
                                            temperature=self.temperature,
                                            min_new_tokens=3,
                                            max_new_tokens=80,
                                            eos_token_id='\n').text
        print("#" * 25 + "ANSWER Output" + "#" * 25)
        print(answer)

        confidence = 0.8
        result = AnswerResult("ANSWER", self.question, answer, confidence)
        return result

class Hypothesis():
    """
    A class to generate answers for given prompts using a machine learning model.
    
    Attributes:
        base_model (BaseModel): The machine learning model used to generate answers.
        temperature (float): The sampling temperature to use when generating answers.
        question (str): The question for which the answer is generated.
    """

    def __init__(self, base_model, temperature, example) -> None:
        """
        Constructs all the necessary attributes for the AnswerGenerator object.
        
        Parameters:
            base_model (BaseModel): The pre-trained base model for generating answers.
            temperature (float): A coefficient to control the randomness of predictions
                                 by scaling the logits before applying softmax.
            example (dict): A dictionary containing a 'question' key.
        """
        self.base_model = base_model
        self.temperature = temperature
        self.question = example['Q']

    def comment(self, prompt, state: str, details) -> str:
        """
        Generates an answer for the given prompt.

        Parameters:
            prompt (str): The prompt to which the model should respond.
            state (dict): The current state information to be considered by the model.

        Returns:
            AnswerResult: An object containing the action, question, generated answer, and confidence level.
        """
        
        model_input = utils.hypothesis_prompt(prompt, self.question, state, "HYPOTHESIS", details)
        print("#" * 25 + "HYPOTHESIS Input" + "#" * 25)
        print(model_input)

        answer = self.base_model.generate([model_input],
                                            hide_input=True,
                                            do_sample=True,
                                            temperature=self.temperature,
                                            min_new_tokens=3,
                                            eos_token_id='\n').text
        print("#" * 25 + "HYPOTHESIS Output" + "#" * 25)
        print(answer)

        confidence = 0.8
        #result = AnswerResult("ANSWER", self.question, answer, confidence)
        result = HypothesisResult(
            state_type = "HYPOTHESIS",
            proposition = details,
            comment = answer,
            confidence = confidence
        )
        return result


class Toolbox():
    def __init__(self, world_model) -> None:

        self.world_model = world_model
        self.example = world_model.example
        self.prompt = world_model.prompt

        self.retrieval = Retrieval(
            example = self.example,
            hyparams = self.prompt["actions"]["RETRIEVE"]["hyparams"]
        )
        self.answer = Answer(
            base_model=self.world_model.base_model,
            temperature=self.world_model.temperature,
            example = self.example
        )
        self.hypothesize = Hypothesis(
            base_model=self.world_model.base_model,
            temperature=self.world_model.temperature,
            example = self.example
        )
        self.keywords = ['ANSWER', 'DECOMPOSE', 'RETRIEVE', 'INVALID', 'HYPOTHESIS']

    def execute_tool(self, prompt, example, state, action: str) -> str:
        #keyword = utils.find_first_appearance(action, self.keywords)
        keyword, details = action

        if keyword == 'ANSWER':
            return self.answer.answer(prompt, state)
        elif keyword == 'RETRIEVE':
            #return self.retrieval.retrieve(action)
            return self.retrieval.retrieve(state, details)
        elif keyword == 'HYPOTHESIS':
            #return self.retrieval.retrieve(action)
            return self.hypothesize.comment(prompt, state, details)
        elif keyword == 'INVALID':
            return InvalidResult("INVALID", "INVALID")
        else:
            raise KeyError(f"Action {keyword} not found in {self.keywords}")



class WebQAWorldModel(WorldModel[GSM8kState, WebQAAction]):
    """
    GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_confidence=8,
                 batch_size=2,
                 temperature=0.8,
                 early_stop_base=None,
                 early_stop_threshold=1.) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt: GSM8kPrompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold

    def init_state(self) -> list:
        return []

    def update_example(self, example: str) -> None:
        super().update_example(example)
        self.tools = Toolbox(self)


    def step(self, state: GSM8kState, action: WebQAAction) -> tuple[GSM8kState, dict]:
        state = state.copy()

        result = self.tools.execute_tool(self.prompt, self.example, state, action)

        state.append(result)

        # TODO take care of aux later
        aux = {'confidence': 0.8}
        return state, aux

    def is_terminal(self, state: GSM8kState) -> bool:
        # check when used namedtuples
        # if len(state) > 0 and type(state[-1]).__name__ == 'ANSWER':
        if len(state) > 0 and (state[-1].state_type == 'ANSWER' or state[-1].state_type == 'INVALID'):
            return True
        else:
            return False
