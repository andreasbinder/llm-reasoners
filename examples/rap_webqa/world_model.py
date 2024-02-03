import io
from typing import NamedTuple, TypedDict, Union, Tuple, List
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils

import json

from tools import RetrievalResult, Query, get_retriever # Retrieval,
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

    def __init__(self, base_model, temperature, example, hyparams) -> None:
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
        self.hyparams = hyparams
        self.min_new_tokens = hyparams["min_new_tokens"]
        self.max_new_tokens = hyparams["max_new_tokens"]

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
                                            min_new_tokens=self.min_new_tokens,
                                            max_new_tokens=self.max_new_tokens,
                                            eos_token_id='\n').text
        
        answer = [a.strip() for a in answer]
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
        
        #model_input = utils.hypothesis_prompt(prompt, self.question, state, "HYPOTHESIS", details)
        model_input = utils.state_transition_prompt(prompt, self.question, state, "HYPOTHESIS", details)
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

from sentence_transformers.cross_encoder import CrossEncoder

# class RefineResult(NamedTuple):
#     state_type: str
#     scores: str
#     snippets: str
#     confidence: float

# class Refine():
#     """
#     A class to generate answers for given prompts using a machine learning model.
    
#     Attributes:
#         base_model (BaseModel): The machine learning model used to generate answers.
#         temperature (float): The sampling temperature to use when generating answers.
#         question (str): The question for which the answer is generated.
#     """

#     def __init__(self, base_model, temperature, example) -> None:
#         """
#         Constructs all the necessary attributes for the AnswerGenerator object.
        
#         Parameters:
#             base_model (BaseModel): The pre-trained base model for generating answers.
#             temperature (float): A coefficient to control the randomness of predictions
#                                  by scaling the logits before applying softmax.
#             example (dict): A dictionary containing a 'question' key.
#         """
#         self.base_model = base_model
#         self.temperature = temperature
#         self.question = example['Q']
#         self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')   
#         self.top_k = 2

#     def filter(self, prompt, state: str, details) -> str:
#         """
#         Generates an answer for the given prompt.

#         Parameters:
#             prompt (str): The prompt to which the model should respond.
#             state (dict): The current state information to be considered by the model.

#         Returns:
#             AnswerResult: An object containing the action, question, generated answer, and confidence level.
#         """
        
        
        
#         print("#" * 25 + "REFINE Input" + "#" * 25)
        

#         snippets = []
#         for s in state:
#             if s.state_type == "RETRIEVE":
#                 snippets += s.retrieved_snippets
                
#         if len(snippets) == 0:
#             confidence = 0.8
#             result = RefineResult(
#                 state_type = "REFINE",
#                 scores = [],
#                 snippets = [],
#                 confidence = confidence
#             )
#             return result

#         input_texts = [
#             [self.question, snippet] for snippet in snippets
#         ]


#         #scores = model.predict(input_texts, apply_softmax=False)
#         scores = self.cross_encoder.predict(input_texts, apply_softmax=True)

#         import numpy as np

#         print(scores)

#         # Sort the scores in decreasing order
#         sim_scores_argsort = list(reversed(np.argsort(scores)))[:self.top_k]

#         flags = [True] * len(scores)
#         for idx in sim_scores_argsort:
#             flags[idx] = False
        
#         k = 4
#         index = 0
#         for s in state:
#             if s.state_type == "RETRIEVE":
#                 s.is_filtered = flags[index:index+k]
#                 index += k




#         print("#" * 25 + "REFINE Output" + "#" * 25)
        

#         confidence = 0.8
#         #result = AnswerResult("ANSWER", self.question, answer, confidence)
#         result = RefineResult(
#             state_type = "REFINE",
#             scores = scores,
#             snippets = snippets,
#             confidence = confidence
#         )

#         return result

import numpy as np


class RefineResult(NamedTuple):
    state_type: str
    state_indices: str
    snippet_indices: str
    snippets: str
    sources: str
    scores: float
    confidence: float   

# class RefineResult:
#     def __init__(self, state_type, state_indices, snippets, scores, confidence):
#         self.state_type = state_type
#         self.state_indices = state_indices
#         self.snippets = snippets
#         self.scores = scores
#         self.confidence = confidence

class Refine:
    def __init__(self, base_model, temperature, example, hyparams, top_k=4):
        self.base_model = base_model
        self.temperature = temperature
        self.question = example['Q']
        self.hyparams = hyparams

        self.top_k = self.hyparams['top_k']
        self.crossencoder_model = hyparams['crossencoder_model']['checkpoint']

        self.cross_encoder = CrossEncoder(self.crossencoder_model)   

    # def filter(self, state) -> RefineResult:
    #     snippets = []
    #     state_indices = []

    #     # Retrieve snippets from the state
    #     for idx, s in enumerate(state):
    #         if s.state_type == "RETRIEVE":
    #             snippets.extend(s.retrieved_snippets)
    #             state_indices.extend([idx] * len(s.retrieved_snippets))

    #     if len(snippets) == 0:
    #         return RefineResult("REFINE", [], [], [], 0.8)

    #     input_texts = [[self.question, snippet] for snippet in snippets]
    #     scores = self.cross_encoder.predict(input_texts, apply_softmax=True)

    #     # Get top-k highest scores
    #     top_indices = np.argsort(scores)[-self.top_k:]

    #     # Extract top k snippets and their original state indices
    #     top_snippets = [snippets[i] for i in top_indices]
    #     top_state_indices = [state_indices[i] for i in top_indices]

    #     return RefineResult(
    #         "REFINE", 
    #         top_state_indices, 
    #         top_snippets, 
    #         scores[top_indices].tolist(), 
    #         0.8
    #     )

    def filter(self, state) -> RefineResult:
        snippets = []
        indices_info = []  # Stores tuples of (state_index, snippet_index)
        source_ids = []

        # Retrieve snippets from the state and keep track of their indices
        for state_index, s in enumerate(state):
            if s.state_type == "RETRIEVE":
                for snippet_index, snippet in enumerate(s.retrieved_snippets):
                    if not s.is_filtered[snippet_index]:
                        snippets.append(snippet)
                        indices_info.append((state_index, snippet_index))
                    source_ids.extend(s.retrieved_sources)

        if len(snippets) == 0:
            return RefineResult("REFINE", [], [], [], [], [], 0.8)

        input_texts = [[self.question, snippet] for snippet in snippets]
        scores = self.cross_encoder.predict(input_texts, apply_softmax=True)

        # Get top-k highest scores
        top_indices = np.argsort(scores)[-self.top_k:]

        # Extract top k snippets and their indices information
        top_snippets = [snippets[i] for i in top_indices]
        top_indices_info = [indices_info[i] for i in top_indices]

        # Separate state indices and snippet indices for RefineResult
        top_state_indices, top_snippet_indices = zip(*top_indices_info)

        top_sources = [source_ids[i] for i in top_indices]

        print("#" * 25 + "REFINE Input" + "#" * 25)
        print(list(top_state_indices))
        print(list(top_snippet_indices))
        print(top_snippets)
        print(top_sources)


    
        print("#" * 25 + "REFINE Output" + "#" * 25)

        return RefineResult(
            "REFINE", 
            list(top_state_indices), 
            list(top_snippet_indices),
            top_snippets, 
            top_sources, 
            scores[top_indices].tolist(), 
            0.8
        )


# Example usage
# refine = Refine(base_model, temperature, example)
# result = refine.filter(state)


class Toolbox():
    def __init__(self, world_model) -> None:

        self.world_model = world_model
        self.example = world_model.example
        self.config = world_model.config

        

        if "RETRIEVE" in self.config["action_selection"]["available_actions"]:
            

            retrieve_hyparams = self.config["actions"]["RETRIEVE"]["hyparams"]
            Retrieval = get_retriever(retrieve_hyparams['embedding_model']['type'])
            self.retrieval = Retrieval(
                example = self.example,
                hyparams = retrieve_hyparams
            )

        if "ASPECT" in self.config["action_selection"]["available_actions"]:
            
            retrieve_hyparams = self.config["actions"]["ASPECT"]["hyparams"]
            Retrieval = get_retriever(retrieve_hyparams['embedding_model']['type'])
            self.retrieval = Retrieval(
                example = self.example,
                hyparams = retrieve_hyparams
            )

        if "QUERY" in self.config["action_selection"]["available_actions"]:
            self.query = Query(
                example = self.example,
                hyparams = self.config["actions"]["QUERY"]["hyparams"]
            )

        if "ANSWER" in self.config["action_selection"]["available_actions"] or self.world_model.final_state_type:    
            self.answer = Answer(
                base_model=self.world_model.base_model,
                temperature=self.world_model.temperature,
                example = self.example,
                hyparams = self.config["actions"]["ANSWER"]["hyparams"]
            )

        if "HYPOTHESIS" in self.config["action_selection"]["available_actions"]:        
            self.hypothesize = Hypothesis(
                base_model=self.world_model.base_model,
                temperature=self.world_model.temperature,
                example = self.example
            )

        if "REFINE" in self.config["action_selection"]["available_actions"]:
            refine_hyparams = self.config["actions"]["REFINE"]["hyparams"]
            self.refine = Refine(
                base_model=self.world_model.base_model,
                temperature=self.world_model.temperature,
                example = self.example,
                hyparams=refine_hyparams
            )
        self.keywords = ['ANSWER', 'DECOMPOSE', 'RETRIEVE', 'INVALID', 'HYPOTHESIS']
        

    def execute_tool(self, prompt, example, state, action: str) -> str:
        #keyword = utils.find_first_appearance(action, self.keywords)
        keyword, details = action

        if keyword == 'ANSWER':
            return self.answer.answer(prompt, state)
        elif keyword == 'RETRIEVE' or keyword == 'ASPECT':
            #return self.retrieval.retrieve(action)
            return self.retrieval.retrieve(state, details)
        elif keyword == 'QUERY':
            #return self.retrieval.retrieve(action)
            return self.query.retrieve(state, details)
        elif keyword == 'HYPOTHESIS':
            #return self.retrieval.retrieve(action)
            return self.hypothesize.comment(prompt, state, details)
        elif keyword == 'REFINE':
            #return self.retrieval.retrieve(action)
            return self.refine.filter(state)
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
        self.config: GSM8kPrompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold

        self.final_state_type = self.config['general']['final_state_type']

    def init_state(self) -> list:
        if not self.config['general']['use_bootstrap']: 
            return []
        
        else:
            return self.bootstrap_state(self.example)

    def bootstrap_state(self, example):
        from utils import fetch_images_by_id
        state = []
        path = self.config['bootstrap']['bootstrap_path']
        with open(path, 'r') as f:
            json_data = json.load(f)

        webqa_path = example['path']
        with open(webqa_path, 'r') as f:
            webqa_data = json.load(f)

        guid = example["Guid"]
        data = json_data[guid]
        webqa_data = webqa_data[guid]

        retrieved_snippets=[]
        retrieved_sources=[]
        modalities=[]
        is_filtered =[]
        #n_sources = len(data['sources'])

        for source in data['sources']:
            if type(source) == str:
                for snippet in webqa_data['txt_Facts']:
                    if snippet['snippet_id'] == source:
                        retrieved_snippets.append(snippet['fact'])
                        break
                        # retrieved_sources.append(source)
                        # modalities.append('text')
                #text = webqa_data['txt_Facts']
                #retrieved_snippets.append(source)
                retrieved_sources.append(source)
                modalities.append('text')
                is_filtered.append(False)
            if type(source) == int:
                
                image = fetch_images_by_id(
                    [source],
                    self.tools.retrieval.lineidx_file,
                    self.tools.retrieval.tsv_file
                    )
                
                for snippet in webqa_data['img_Facts']:
                    if snippet['image_id'] == source:
                        #retrieved_snippets.append(snippet['fact'])
                        image_caption = snippet['caption']
                        break

                # Generate a caption for the image based on the query
                caption_model_prompt = self.config['bootstrap']['bootstrap_prompt'].format(
                    overall_question=example['Q'],
                    caption=image_caption
                )

                print("#" * 25 + "Start" + str(source) + "#" * 25)
                print(caption_model_prompt)
                print("#" * 25 )
                
                        

                caption = self.tools.retrieval.caption_model.generate_caption(prompt=caption_model_prompt, 
                                                   image_file=image, 
                                                   max_new_tokens=self.config['bootstrap']['hyparams']['max_new_tokens'],
                                                   min_new_tokens=self.config['bootstrap']['hyparams']['min_new_tokens'])  # Assuming empty text prompt
                print(caption)
                print("#" * 25 )
                
                

                caption = self.tools.retrieval.sanitize_vlm_output(caption)
                #caption = self.generate_image_caption(query, image)
                print(caption)
                print("#" * 25 + "End" + str(source) + "#" * 25)
                # if 'IGNORE' in caption:
                #     continue

                if self.config['bootstrap']['filter_token'] in caption:
                    is_filtered.append(True)
                else:
                    is_filtered.append(False)

                retrieved_snippets.append(caption)
                retrieved_sources.append(source)
                modalities.append('image')


        state.append(
            RetrievalResult(
                state_type="RETRIEVE",
                context='bootstrap',
                retrieved_snippets=retrieved_snippets,
                retrieved_sources=retrieved_sources,
                modalities=modalities,
                relevance_scores=[0.] * len(modalities),
                is_gold=[False] * len(modalities),
                is_filtered=is_filtered,
            )
        )
        return state

    # def bootstrap_state(self, example):
    #     from utils import fetch_images_by_id
    #     state = []
    #     path = self.config['bootstrap']['bootstrap_path']
    #     with open(path, 'r') as f:
    #         json_data = json.load(f)

    #     webqa_path = example['path']
    #     with open(webqa_path, 'r') as f:
    #         webqa_data = json.load(f)

    #     guid = example["Guid"]
    #     data = json_data[guid]
    #     webqa_data = webqa_data[guid]

    #     retrieved_snippets=[]
    #     retrieved_sources=[]
    #     modalities=[]
    #     is_filtered =[]
    #     is_gold =[]
    #     #n_sources = len(data['sources'])

    #     for source in data['sources']:
    #         if type(source) == str:
    #             for snippet in webqa_data['txt_posFacts']:
    #                 if snippet['snippet_id'] == source:
    #                     retrieved_snippets.append(snippet['fact'])
    #                     break
    #                     # retrieved_sources.append(source)
    #                     # modalities.append('text')
    #             #text = webqa_data['txt_Facts']
    #             #retrieved_snippets.append(source)
    #             retrieved_sources.append(source)
    #             modalities.append('text')
    #             is_filtered.append(False)
    #             is_filtered.append(True)
    #         if type(source) == str:
    #             for snippet in webqa_data['txt_negFacts']:
    #                 if snippet['snippet_id'] == source:
    #                     retrieved_snippets.append(snippet['fact'])
    #                     break
    #                     # retrieved_sources.append(source)
    #                     # modalities.append('text')
    #             #text = webqa_data['txt_Facts']
    #             #retrieved_snippets.append(source)
    #             retrieved_sources.append(source)
    #             modalities.append('text')
    #             is_filtered.append(False)
    #             is_filtered.append(False)
    #         if type(source) == int:
                
    #             image = fetch_images_by_id(
    #                 [source],
    #                 self.tools.retrieval.lineidx_file,
    #                 self.tools.retrieval.tsv_file
    #                 )
                
    #             for snippet in webqa_data['img_Facts']:
    #                 if snippet['image_id'] == source:
    #                     #retrieved_snippets.append(snippet['fact'])
    #                     image_caption = snippet['caption']
    #                     break

    #             # Generate a caption for the image based on the query
    #             caption_model_prompt = self.config['bootstrap']['bootstrap_prompt'].format(
    #                 overall_question=example['Q'],
    #                 caption=image_caption
    #             )

    #             print("#" * 25 + "Start" + str(source) + "#" * 25)
    #             print(caption_model_prompt)
    #             print("#" * 25 )
                
                        

    #             caption = self.tools.retrieval.caption_model.generate_caption(prompt=caption_model_prompt, 
    #                                                image_file=image, 
    #                                                max_new_tokens=self.config['bootstrap']['hyparams']['max_new_tokens'],
    #                                                min_new_tokens=self.config['bootstrap']['hyparams']['min_new_tokens'])  # Assuming empty text prompt
    #             print(caption)
    #             print("#" * 25 )
                
                

    #             caption = self.tools.retrieval.sanitize_vlm_output(caption)
    #             #caption = self.generate_image_caption(query, image)
    #             print(caption)
    #             print("#" * 25 + "End" + str(source) + "#" * 25)
    #             # if 'IGNORE' in caption:
    #             #     continue

    #             if self.config['bootstrap']['filter_token'] in caption:
    #                 is_filtered.append(True)
    #             else:
    #                 is_filtered.append(False)

    #             retrieved_snippets.append(caption)
    #             retrieved_sources.append(source)
    #             modalities.append('image')


    #     state.append(
    #         RetrievalResult(
    #             state_type="RETRIEVE",
    #             context='bootstrap',
    #             retrieved_snippets=retrieved_snippets,
    #             retrieved_sources=retrieved_sources,
    #             modalities=modalities,
    #             relevance_scores=[0.] * len(modalities),
    #             is_gold=[False] * len(modalities),
    #             is_filtered=is_filtered,
    #         )
    #     )
    #     return state


    def update_example(self, example: str) -> None:
        super().update_example(example)
        self.tools = Toolbox(self)


    def step(self, state: GSM8kState, action: WebQAAction) -> tuple[GSM8kState, dict]:
        state = state.copy()

        result = self.tools.execute_tool(self.config, self.example, state, action)

        state.append(result)

        # TODO take care of aux later
        aux = {'confidence': 0.8}
        return state, aux

    def is_terminal(self, state: GSM8kState) -> bool:
        # check when used namedtuples
        # if len(state) > 0 and type(state[-1]).__name__ == 'ANSWER':
        if len(state) > 0 and (
                state[-1].state_type == self.final_state_type or state[-1].state_type == 'INVALID'
            ):
            return True
        else:
            return False
