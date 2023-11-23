import io
from typing import NamedTuple, TypedDict, Union, Tuple, List
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils

import os
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
from pathlib import Path
import json

# Make sure to define or import RetrievalResult somewhere in your code.

import json
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import base64
from io import BytesIO

class RetrievalResult(NamedTuple):
    state_type: str
    context: str
    retrieved_snippets: List[str]
    retrieved_sources: List[str]
    flags: List[str]
    relevance_scores: List[float]

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

class TextRetrieval():
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

        # TODO less efficient than open
        self.db = json.loads(Path(path).read_text())[guid]
        

        metadata_func_with_extra = self.create_metadata_func(path, index)

        if self.example.get('split', None) == "test": 
            jq_schema=f'.{guid}.txt_Facts[]'
            self.set_all_snippet_ids = set(snippet['snippet_id'] for snippet in self.db['txt_Facts'])
        else:
            jq_schema=f'.{guid}.txt_posFacts[], .{guid}.txt_negFacts[]'
            self.set_all_snippet_ids = set(snippet['snippet_id'] for snippet in self.db['txt_posFacts']) | set(snippet['snippet_id'] for snippet in self.db['txt_negFacts'])

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


# import os
# #import clip
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch


# import os
# import json
# from pathlib import Path
# #import clip
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch
# from utils.webqa_helpers import load_line_indices, batch_image_id_to_images, fetch_images_by_id
# from sentence_transformers import util
# #from models import InstructBlip
# from transformers import CLIPModel, CLIPProcessor
# import torch

# class ClipRetrieval():
#     def __init__(self, example, hyparams):
#         # Extracting parameters from hyparams
#         self.mode = hyparams.get('mode', 'mm')  # Default to multimodal
#         device = hyparams.get('device', 'cuda')
#         checkpoint = hyparams.get('clip_checkpoint', None) #Assuming checkpoint is mandatory
#         generation_config = hyparams.get('generation_config', {})
#         bnb_config = hyparams.get('bnb_config', {})
#         lineidx_file = hyparams.get('lineidx_file', '/nfs/data2/zhangya/webqa/imgs.lineidx')
#         tsv_file = hyparams.get('tsv_file', '/nfs/data2/zhangya/webqa/imgs.tsv')


#         self.device = device
#         #self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
#         # self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#         # self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.model.to(self.device)
#         from unittest.mock import MagicMock

#         # Create a mock InstructBlip object
#         self.instruct_blip = MagicMock()
#         self.instruct_blip.generate_caption.return_value = "A mock caption for the image."
#         self.embeddings = {}  # Dictionary to store embeddings
#         self.source_material = {}  # Dictionary to store source material

#         self.example = example
#         self.lineidx_file = lineidx_file
#         self.tsv_file = tsv_file
#         self.load_data()

#     def load_data(self):
#         path = self.example['path']
#         guid = self.example['Guid']
#         self.db = json.loads(Path(path).read_text())[guid]

#         for fact_type in ['txt_posFacts', 'txt_negFacts', 'img_posFacts', 'img_negFacts']:
#             if 'txt' in fact_type:
#                 for item in self.db.get(fact_type, []):
#                     text = item['fact']
#                     id = item['snippet_id']  # Assuming each item has a unique snippet_id
#                     self.add_to_index(id, text=text)
#             elif 'img' in fact_type:
#                 image_ids = [item['image_id'] for item in self.db.get(fact_type, [])]
#                 images = fetch_images_by_id(image_ids, self.lineidx_file, self.tsv_file)
#                 for image_id, image in zip(image_ids, images):
#                     self.add_to_index(image_id, image_path=image)

#     def add_to_index(self, id, text=None, image_path=None):
#         if text:
#             text_emb = self.embed_text(text)
#             self.embeddings[id] = {'type': 'text', 'embedding': text_emb}
#         if image_path:
#             image_emb = self.embed_image(image_path)
#             self.embeddings[id] = {'type': 'image', 'embedding': image_emb}
        
#         if text:
#             self.source_material[id] = {'type': 'text', 'content': text}
#         if image_path:
#             self.source_material[id] = {'type': 'image', 'content': image_path}

#     def embed_text(self, text):
#         inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         with torch.no_grad():
#             text_features = self.model.get_text_features(**inputs)
#         return text_features / text_features.norm(dim=-1, keepdim=True)

#     def embed_image(self, image_path):
#         image = self.processor(images=image_path, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             image_features = self.model.get_image_features(**image)
#         return image_features / image_features.norm(dim=-1, keepdim=True)

#     def search(self, query, top_k=5):
#         query_emb = self.embed_text(query) # [1,512]
#         #corpus_embeddings = [self.embeddings[key]['embedding'] for key in self.embeddings]
#         corpus_embeddings = torch.stack([self.embeddings[key]['embedding'].squeeze() for key in self.embeddings]) # [34,512]
#         search_results = util.semantic_search(query_emb, corpus_embeddings, top_k=top_k)[0]
#         return search_results

#     def retrieve(self, query):
#         # search_results = self.search(query)
#         # retrieved_snippets = []
#         # retrieved_sources = []
#         # flags = []
#         # relevance_scores = []

#         # for result in search_results:
#         #     corpus_id = result['corpus_id']
#         #     score = result['score']
#         #     id = list(self.embeddings.keys())[corpus_id]
#         #     emb_data = self.embeddings[id]

#         #     retrieved_sources.append(id)
#         #     relevance_scores.append(score)

#         #     if emb_data['type'] == 'text':
#         #         retrieved_snippets.append(emb_data['text'])
#         #         flags.append('text')
#         #     elif emb_data['type'] == 'image':
#         #         image_path = emb_data['image_path']
#         #         caption = self.instruct_blip.generate_caption("", image_path)  # Assuming empty text prompt
#         #         retrieved_snippets.append(caption)
#         #         flags.append('image')

#         search_results = self.search(query)
#         retrieved_snippets = []
#         retrieved_sources = []
#         flags = []
#         relevance_scores = []

#         for result in search_results:
#             corpus_id = result['corpus_id']
#             score = result['score']
#             id = list(self.source_material.keys())[corpus_id]
#             source_data = self.source_material[id]

#             retrieved_sources.append(id)
#             relevance_scores.append(score)

#             if self.mode == 'txt':
#                 # Handle text-only retrieval
#                 retrieved_snippets.append(source_data['content'])
#                 flags.append('text')
#             elif self.mode == 'img':
#                 # Handle image-only retrieval
#                 # For images, you might want to generate captions or just return the image paths
#                 image_path = source_data['content']
#                 caption = self.instruct_blip.generate_caption("", image_path)  # Assuming empty text prompt
#                 retrieved_snippets.append(caption)
#                 flags.append('image')
#             else:
#                 if source_data['type'] == 'text':
#                     retrieved_snippets.append(source_data['content'])
#                     flags.append('text')
#                 elif source_data['type'] == 'image':
#                     image_path = source_data['content']
#                     caption = self.instruct_blip.generate_caption("", image_path)  # Assuming empty text prompt
#                     retrieved_snippets.append(caption)
#                     flags.append('image')

#         result = RetrievalResult(
#             state_type="RETRIEVE",
#             context=query,
#             retrieved_snippets=retrieved_snippets,
#             retrieved_sources=retrieved_sources,
#             flags=flags,
#             relevance_scores=relevance_scores
#         )
#         return result
    
#     def generate_questions(self, query):
#         search_results = self.search(query, top_k=5)
#         questions = []
#         for id, _ in search_results:
#             if self.embeddings[id]['type'] == 'image':
#                 question = self.generate_question_from_image(self.embeddings[id]['path'])
#                 questions.append(question)
#         return questions

#     def generate_question_from_image(self, image_path):
#         image = Image.open(image_path)
#         inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
#         outputs = self.blip_model.generate(**inputs)
#         question = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
#         return question

# # Example Usage
# # retrieval = ClipRetrieval()
# # retrieval.add_to_index('id1', text='A sample text')
# # retrieval.add_to_index('id2', image_path='path_to_image.jpg')
# # results = retrieval.search('query text')
# # questions = retrieval.generate_questions('query text')

# if __name__ == "__main__":
#     # Example usage
#     example = {
#         'path': '/home/stud/abinder/master-thesis/data/n_samples_50_split_val_solution_img_seed_42_1691423195.1279488_samples_dict.json',
#         'Guid': 'd5c4710c0dba11ecb1e81171463288e9'
#     }
#     retrieval = ClipRetrieval(example, {})
#     result = retrieval.retrieve('query text')