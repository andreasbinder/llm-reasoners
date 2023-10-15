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
DecomposeResult = namedtuple("DECOMPOSE", ["sub_question", "sub_answer", "confidence"])

RetrievalResult = namedtuple("RETRIEVE", ["context", "retrieved_snippets", "retrieved_sources"])

AnswerResult = namedtuple("ANSWER", ["main_question", "main_answer", "confidence"])

GSM8kState = list[Union[RetrievalResult, DecomposeResult, AnswerResult]]
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
        
        path = '/home/stud/abinder/Multimodal-LLMs-for-webscale-Questions-Answering/data/n_samples_50_split_val_solution_txt_seed_42_1691423190.7960498_samples.json'
        
        loader = JSONLoader(
            file_path=path,
            jq_schema='.[0].txt_posFacts[], .[0].txt_negFacts[]',
            content_key="fact",
            text_content=True,
        )
        documents = loader.load()      

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}

        from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

        # key should be inferred from bashrc
        embeddings_api = HuggingFaceInferenceAPIEmbeddings(
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
        docs = self.vectorstore.similarity_search(query)
        return docs
        

        

class Toolbox():
    def __init__(self) -> None:
        self.retrieval = Retrieval()

    def execute_tool(self, question: str) -> str:
        return self.retrieval.retrieve(question)





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
        self.tools = Toolbox()

    def step(self, state: GSM8kState, action: GSM8kAction) -> tuple[GSM8kState, dict]:
        state = state.copy()

        docs = self.tools.execute_tool(action)


        state.append(RetrievalResult(action, ','.join([f'{i}) ' + doc.page_content \
                                            for i, doc in enumerate(docs)]), [[doc.metadata['source'] for doc in docs]]))

        with io.StringIO() as f:
            f.write(self.prompt["input"])
            f.write(self.prompt["question_prefix"] + self.example + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(self.prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.prompt["answer_prefix"].format(len(state) + 1))
            model_input = f.getvalue()

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num = stop - start

                outputs = self.base_model.generate([model_input] * num,
                                                   hide_input=True,
                                                   do_sample=True,
                                                   temperature=self.temperature,
                                                   eos_token_id='\n').text
                for output in outputs:
                    result = output.strip()
                    answer = utils.retrieve_answer(result)
                    if answer is not None:
                        answer_dict[answer].append(result)

            # Early stop if confidence is high enough
            if len(answer_dict) == 0:  # no answer yet
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 2 and max_len == len(sorted_answer_dict[1][1]):
                    pass  # Tie with the second best answer
                else:
                    break

        if len(answer_dict) == 0:
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        state.append(SubResult(action, answer, confidence))
        aux = {'confidence': confidence}
        return state, aux

    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False
