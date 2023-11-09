import io
from typing import NamedTuple, TypedDict, Union, Tuple, List
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
    retrieved_snippets: List[str]
    retrieved_sources: List[str]
    flags: List[str]
    relevance_scores: List[float]
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
        
import os
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
from pathlib import Path
import json

# Make sure to define or import RetrievalResult somewhere in your code.

class WebQAVectorStore(FAISS):
    SCORING_MODE = None
    # def __init__(self, documents, embeddings, scoring_mode, *args, **kwargs):
    #     self.scoring_mode = scoring_mode 
    #     super().__init__(documents, embeddings, *args, **kwargs)

    @classmethod
    def from_documents(cls, documents, embeddings, scoring_mode, *args, **kwargs):
        # Perform any preprocessing required for the documents or embeddings
        # ...

        # Assuming 'SuperClass' is the name of your parent class and you want to
        # call 'from_documents' of the superclass, you can do it like this:
        instance = super(cls, cls).from_documents(documents, embeddings, *args, **kwargs)

        # Now that you have the instance, you can set additional properties or do more with it
        instance.scoring_mode = scoring_mode
        # Do anything else with the instance as needed
        
        return instance


    # @classmethod 
    # def from_documents(cls, documents, embeddings, scoring_mode, *args, **kwargs):
    #     # Perform any preprocessing required for the documents or embeddings
    #     # For example, we could normalize embeddings if specified to do so:
    #     # if kwargs.get('normalize_L2', False):
    #     #     embeddings = cls._normalize_embeddings(embeddings)
        
    #     # Any additional preprocessing steps can be added here
    #     global SCORING_MODE 
    #     SCORING_MODE = scoring_mode
    #     # Then call the constructor with the preprocessed data and other arguments
    #     return cls(documents, embeddings, *args, **kwargs)

    def _cosine_range_adjusted(self, score):
            # transformed_similarity = (cosine_similarity + 1) / 2
            return (self._cosine_relevance_score_fn(score) + 1) / 2

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None, k_fetch=20):
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, fetch_k=k_fetch, 
        )
            
        # self.normalized_scores = [self._cosine_range_adjusted(score) for _, score in docs_and_scores]
        # return [(doc, score) for doc, score in docs_and_scores]
    
        if self.scoring_mode == 'cosine':
            normalized_scores = [self._cosine_range_adjusted(score) for _, score in docs_and_scores]
        elif self.scoring_mode == 'euclidean':
            normalized_scores = [self._euclidean_relevance_score_fn(score) for _, score in docs_and_scores]
        else:
            raise ValueError(f"Scoring mode {self.scoring_mode} not supported.")

        docs, relevance_scores = zip(*docs_and_scores)
        # Return docs with normalized scores
        return [(doc, score) for doc, score in zip(docs, normalized_scores)]




class Retrieval():
    def __init__(self, example, hyparams):
        """
        Initialize the Retrieval class with given example configuration.
        :param example: Example configuration for retrieval.
        :param use_api: A flag to indicate whether to use the HuggingFace API or local embeddings.
        """
        self.example = example
        self.use_api = hyparams.get('use_api', False)
        self.use_cache = hyparams.get('use_cache', True)
        self.vector_store_kwargs = hyparams.get('vector_store_kwargs', {
            "normalize_L2": False,
            "scoring_mode": "euclidean"
        })
        self.search_kwargs = hyparams.get('search_kwargs', {
            "k": 4,
            "k_fetch": 20
        })
        # Note: that sentence-transformers/all-mpnet-base-v2 considered best allarounder mode
        # Probably multi-qa-distilbert-cos-v1 is better for cosine similarity -> every model was optimized differently
        self.model_name = hyparams.get('model_name', "sentence-transformers/all-mpnet-base-v2")
        self.model_kwargs = hyparams.get('model_kwargs', {
            "device": "cuda"
        })
        self.encode_kwargs = hyparams.get('encode_kwargs', {
            "normalize_embeddings": False
        })

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
        index = None
        guid = self.example['Guid']

        self.db = json.loads(Path(path).read_text())[guid]
        self.set_all_snippet_ids = set(snippet['snippet_id'] for snippet in self.db['txt_posFacts']) | set(snippet['snippet_id'] for snippet in self.db['txt_negFacts'])

        metadata_func_with_extra = self.create_metadata_func(path, index)

        if self.example.get('split', None) == "test": 
            jq_schema=f'.{guid}.txt_Facts[]'
        else:
            jq_schema=f'.{guid}.txt_posFacts[], .{guid}.txt_negFacts[]'

        self.loader = JSONLoader(
            file_path=path,
            jq_schema=jq_schema,
            content_key="fact",
            text_content=True,
            metadata_func=metadata_func_with_extra
        )
        self.documents = self.loader.load()

    def setup_embeddings(self):
        """
        Setup the embeddings for the document retrieval.
        """

        if self.use_api:
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=self.huggingface_token,  # Use the stored token
                model_name=self.model_name
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs,
                encode_kwargs=self.encode_kwargs
            )

    def setup_vectorstore(self):
        """
        Setup the vector store for storing and retrieving document embeddings.
        """
        from langchain.vectorstores.utils import DistanceStrategy
        # self.vectorstore = FAISS.from_documents(
        #     self.documents, 
        #     self.embeddings,
        #     #distance_strategy = DistanceStrategy.COSINE
        #     )
        self.vectorstore = WebQAVectorStore.from_documents(
            self.documents, 
            self.embeddings,
            scoring_mode=self.vector_store_kwargs['scoring_mode'],  # Add the scoring_mode parameter
            normalize_L2=self.vector_store_kwargs['normalize_L2'] ,
            distance_strategy = DistanceStrategy.COSINE if self.vector_store_kwargs['scoring_mode'] == 'cosine' else DistanceStrategy.EUCLIDEAN_DISTANCE 
            )

    def create_metadata_func(self, path, index):
        """
        Create a metadata function that adds additional metadata to the records.
        """
        # parent = json.loads(Path(path).read_text())[index]
        parent = self.db

        def metadata_func(record: dict, metadata: dict):
            """
            Add 'pos' or 'neg' flag to the metadata based on the snippet category.
            """
            snippet_id = record.get("snippet_id")
            flag = None

            # Search in positive facts for a matching snippet_id
            for pos_record in parent.get('txt_posFacts', []):
                if pos_record['snippet_id'] == snippet_id:
                    flag = 'pos'
                    break

            # If not found in positive facts, search in negative facts
            if flag is None:
                for neg_record in parent.get('txt_negFacts', []):
                    if neg_record['snippet_id'] == snippet_id:
                        flag = 'neg'
                        break
            
            metadata["flag"] = flag
            metadata["snippet_id"] = snippet_id
            return metadata

        return metadata_func

    def format_retrieval_result(self, query, docs_with_scores):
        """
        Format the retrieval results and create a RetrievalResult object.
        """
        #content_str = ','.join([f'{i}) ' + doc.page_content for i, doc in enumerate(docs)])
        docs, relevance_scores = zip(*docs_with_scores)
        retrieved_snippets = [doc.page_content for doc in docs]
        snippet_ids = [doc.metadata['snippet_id'] for doc in docs]
        flags = [doc.metadata['flag'] for doc in docs]

        result = RetrievalResult(
            state_type = "RETRIEVE",
            context = query, 
            retrieved_snippets = retrieved_snippets, # TODO content_str, 
            retrieved_sources = snippet_ids, 
            flags = flags,
            relevance_scores = relevance_scores
        )
        return result

    def get_unseen_snippet_ids(self, state):
        """
        Get the snippet IDs that have not been seen yet.
        """
        seen_snippet_ids = set()
        for s in state:
            if s.state_type == 'RETRIEVE':
                seen_snippet_ids.update(s.retrieved_sources)
        return self.set_all_snippet_ids - seen_snippet_ids

    def cache_function(self, state):

        if not self.use_cache:
            return None
        
        unseen_snippet_ids = self.get_unseen_snippet_ids(state)
        unseen_snippet_ids_list = list(unseen_snippet_ids)
        return dict(snippet_id=unseen_snippet_ids_list)


    def retrieve(self, state, query: str):

        print("#" * 25 + "RETRIEVE Input" + "#" * 25)
        print(query)
        
        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, 
            k = self.search_kwargs.get('k'), # default 4
            filter=self.cache_function(state),
            k_fetch = self.search_kwargs.get('k_fetch'), # default 20
            )
        
        result = self.format_retrieval_result(query, docs_with_scores)

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
        keyword, details = action

        if keyword == 'ANSWER':
            return self.answer.answer(prompt, state)
        elif keyword == 'RETRIEVE':
            #return self.retrieval.retrieve(action)
            return self.retrieval.retrieve(state, details)
        elif keyword == 'INVALID':
            return InvalidResult("INVALID", "INVALID")
        else:
            raise KeyError(f"Action {keyword} not found in {self.keywords}")



class GSM8kWorldModel(WorldModel[GSM8kState, WebQAAction]):
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
