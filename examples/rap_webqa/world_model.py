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
    retrieved_sources: float
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

class Retrieval():
    def __init__(self) -> None:
        from langchain.document_loaders import JSONLoader
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        
        # path = '/home/stud/abinder/Multimodal-LLMs-for-webscale-Questions-Answering/data/n_samples_50_split_val_solution_txt_seed_42_1691423190.7960498_samples.json'
        path = 'data/n_samples_50_split_val_solution_txt_seed_42_1691423190.7960498_samples.json'

        loader = JSONLoader(
            file_path=path,
            jq_schema='.[1].txt_posFacts[], .[1].txt_negFacts[]',
            content_key="fact",
            text_content=True,
        )
        documents = loader.load()      

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}

        from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings


        import os
        # key should be inferred from bashrc
        HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
        if HUGGINGFACE_TOKEN is None:
            raise ValueError("HUGGINGFACE_TOKEN not set, please run `export HUGGINGFACE_TOKEN=<your key>` to ser it")
       
        embeddings_api = HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGINGFACE_TOKEN,
            model_name=model_name
        )

        #embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        embeddings = embeddings_api
        # storing embeddings in the vector store
        vectorstore = FAISS.from_documents(documents, embeddings)

        self.documents = documents
        self.embeddings = embeddings
        self.vectorstore = vectorstore


    def retrieve(self, query: str) -> str:
        #query = "Where are the best tree travelers in the animal kingdom found in relation to the Salween River in Myanmar?"
        query = query.replace("RETRIEVE: ", "", 1)

        print("#" * 25 + "RETRIEVE Input" + "#" * 25)
        print(query)
        
        docs = self.vectorstore.similarity_search(query)

        result = RetrievalResult(
            "RETRIEVE",
            query, ','.join([f'{i}) ' + doc.page_content \
                                            for i, doc in enumerate(docs)]), [doc.metadata['source'] for doc in docs])

        print("#" * 25 + "RETRIEVE Output" + "#" * 25)
        print(','.join([f'{i}) ' + doc.page_content \
                                            for i, doc in enumerate(docs)]))


        return result
        

class Answer():
    def __init__(self, base_model, temperature) -> None:
        self.base_model = base_model
        self.temperature = temperature

    def answer(self, prompt, example, state, action: str) -> str:

        
        model_input = utils.answer_prompt(prompt, example, state, "ANSWER")
        print("#" * 25 + "ANSWER Input" + "#" * 25)
        print(model_input)

        num = 1
        answer = self.base_model.generate([model_input] * num,
                                            hide_input=True,
                                            do_sample=True,
                                            temperature=self.temperature,
                                            eos_token_id='\n').text
        print("#" * 25 + "ANSWER Output" + "#" * 25)
        print(answer)

        confidence = 0.8
        result = AnswerResult("ANSWER", example, answer, confidence)
        return result

class Toolbox():
    def __init__(self, world_model) -> None:

        self.world_model = world_model

        self.retrieval = Retrieval()
        self.answer = Answer(
            base_model=self.world_model.base_model,
            temperature=self.world_model.temperature
        )
        self.keywords = ['ANSWER', 'DECOMPOSE', 'RETRIEVE', 'INVALID']

    def execute_tool(self, prompt, example, state, action: str) -> str:
        keyword = utils.find_first_appearance(action, self.keywords)
        
        if keyword == 'ANSWER':
            return self.answer.answer(prompt, example, state, action)
        elif keyword == 'RETRIEVE':
            return self.retrieval.retrieve(action)
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

        self.init_tools()

    def init_state(self) -> list:
        return []
    
    def init_tools(self):
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
