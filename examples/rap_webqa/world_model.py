import io
from typing import NamedTuple, TypedDict, Union
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float

# class RetrievalResult(NamedTuple):
#     context: str
#     retrieved_sources: list[str]
#     retrieved_snippets: list[str]

# class DecomposeResult(NamedTuple):
#     sub_question: str
#     sub_answer: str
#     confidence: float
from collections import namedtuple

class DecomposeResult(NamedTuple):
    state_type: str
    sub_question: str
    sub_answer: str
    confidence: float

# DecomposeResult = namedtuple("DECOMPOSE", ["sub_question", "sub_answer", "confidence"])

class RetrievalResult(NamedTuple):
    state_type: str
    context: str
    retrieved_snippets: str
    retrieved_sources: str
    flags: str
# RetrievalResult = namedtuple("RETRIEVE", ["context", "retrieved_snippets", "retrieved_sources"])

class AnswerResult(NamedTuple):
    state_type: str
    main_question: str
    main_answer: str
    confidence: float
# AnswerResult = namedtuple("ANSWER", ["main_question", "main_answer", "confidence"])

class InvalidResult(NamedTuple):
    state_type: str
    main_answer: str
    


GSM8kState = list[Union[RetrievalResult, DecomposeResult, AnswerResult, InvalidResult]]
GSM8kAction = str


class GSM8kPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    answer_prefix: str
    overall_question_prefix: str
        
import os
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
from pathlib import Path
import json

# Make sure to define or import RetrievalResult somewhere in your code.

class Retrieval():
    def __init__(self, example, hyparams):
        """
        Initialize the Retrieval class with given example configuration.
        :param example: Example configuration for retrieval.
        :param use_api: A flag to indicate whether to use the HuggingFace API or local embeddings.
        """
        self.example = example
        self.use_api = hyparams['use_api'] 
        self.documents = None  # Initialize to None or a sensible default
        self.embeddings = None
        self.vectorstore = None

        self.huggingface_token = self.setup_environment()  # Set up tokens or other configurations
        self.load_documents()     # This will update self.documents
        self.setup_embeddings()   # Set up the embedding mechanism
        self.setup_vectorstore()  # Create the vector store based on loaded documents



    def setup_environment(self):
        """
        Setup environment variables and other configurations.
        """
        if self.use_api:
            huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
            if huggingface_token is None:
                raise ValueError(
                    "HUGGINGFACE_TOKEN not set. Please set it in your environment variables."
                )
            return huggingface_token
        return None

    def load_documents(self):
        """
        Load documents from a JSON file based on the given jq schema.
        """
        path = self.example['path']
        index = self.example['index']

        metadata_func_with_extra = self.create_metadata_func(path, index)
        self.loader = JSONLoader(
            file_path=path,
            jq_schema=f'.[{index}].txt_posFacts[], .[{index}].txt_negFacts[]',
            content_key="fact",
            text_content=True,
            metadata_func=metadata_func_with_extra
        )
        self.documents = self.loader.load()

    def setup_embeddings(self):
        """
        Setup the embeddings for the document retrieval.
        """
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}  # Assuming you want to use CUDA for local embedding calculation

        if self.use_api:
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=self.huggingface_token,  # Use the stored token
                model_name=model_name
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs
            )

    def setup_vectorstore(self):
        """
        Setup the vector store for storing and retrieving document embeddings.
        """
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)

    def create_metadata_func(self, path, index):
        """
        Create a metadata function that adds additional metadata to the records.
        """
        parent = json.loads(Path(path).read_text())[index]

        def metadata_func(record: dict, metadata: dict):
            """
            Add 'pos' or 'neg' flag to the metadata based on the snippet category.
            """
            snippet_id = record.get("snippet_id")
            flag = None

            # Search in positive facts for a matching snippet_id
            for pos_record in parent['txt_posFacts']:
                if pos_record['snippet_id'] == snippet_id:
                    flag = 'pos'
                    break

            # If not found in positive facts, search in negative facts
            if flag is None:
                for neg_record in parent['txt_negFacts']:
                    if neg_record['snippet_id'] == snippet_id:
                        flag = 'neg'
                        break
            
            metadata["flag"] = flag
            metadata["snippet_id"] = snippet_id
            return metadata

        return metadata_func

    def format_retrieval_result(self, query, docs):
        """
        Format the retrieval results and create a RetrievalResult object.
        """
        content_str = ','.join([f'{i}) ' + doc.page_content for i, doc in enumerate(docs)])
        snippet_ids = [doc.metadata['snippet_id'] for doc in docs]
        flags = [doc.metadata['flag'] for doc in docs]

        result = RetrievalResult(
            state_type = "RETRIEVE",
            context = query, 
            retrieved_snippets = content_str, 
            retrieved_sources = snippet_ids, 
            flags = flags
        )
        return result

    def retrieve(self, query: str):
        #query = query.replace("RETRIEVE: ", "", 1)

        print("#" * 25 + "RETRIEVE Input" + "#" * 25)
        print(query)
        
        docs = self.vectorstore.similarity_search(query)

        result = self.format_retrieval_result(query, docs)

        print("#" * 25 + "RETRIEVE Output" + "#" * 25)
        print(result.retrieved_snippets)  

        return result


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
        self.question = example['question']

    def answer(self, prompt, state: str) -> str:
        """
        Generates an answer for the given prompt.

        Parameters:
            prompt (str): The prompt to which the model should respond.
            state (dict): The current state information to be considered by the model.

        Returns:
            AnswerResult: An object containing the action, question, generated answer, and confidence level.
        """
        
        model_input = utils.answer_prompt(prompt, self.question, state, "ANSWER")
        print("#" * 25 + "ANSWER Input" + "#" * 25)
        print(model_input)

        answer = self.base_model.generate([model_input],
                                            hide_input=True,
                                            do_sample=True,
                                            temperature=self.temperature,
                                            eos_token_id='\n').text
        print("#" * 25 + "ANSWER Output" + "#" * 25)
        print(answer)

        confidence = 0.8
        result = AnswerResult("ANSWER", self.question, answer, confidence)
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
        self.keywords = ['ANSWER', 'DECOMPOSE', 'RETRIEVE', 'INVALID']

    def execute_tool(self, prompt, example, state, action: str) -> str:
        #keyword = utils.find_first_appearance(action, self.keywords)
        keyword = action[0]

        if keyword == 'ANSWER':
            return self.answer.answer(prompt, state)
        elif keyword == 'RETRIEVE':
            #return self.retrieval.retrieve(action)
            return self.retrieval.retrieve(action[1])
        elif keyword == 'INVALID':
            return InvalidResult("INVALID", "INVALID")
        else:
            raise KeyError(f"Action {keyword} not found in {self.keywords}")



class GSM8kWorldModel(WorldModel[GSM8kState, GSM8kAction]):
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


    def step(self, state: GSM8kState, action: GSM8kAction) -> tuple[GSM8kState, dict]:
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
