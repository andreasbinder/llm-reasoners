import os
#import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


import os
import json
from pathlib import Path
#import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# TODO - Remove this hack
import sys
sys.path.append('/home/stud/abinder/master-thesis')
#from utils.webqa_helpers import load_line_indices, batch_image_id_to_images, fetch_images_by_id
#from utils.webqa_helpers import load_line_indices, batch_image_id_to_images, fetch_images_by_id
####################################################################################
import base64
from PIL import Image
from io import BytesIO

# Function to load line indices
def load_line_indices(file_path):
    with open(file_path, "r") as fp_lineidx:
        return [int(i.strip()) for i in fp_lineidx.readlines()]


# Usage Example
# from utils.webqa_helpers import load_line_indices, batch_image_id_to_images
# lineidx = load_line_indices("/nfs/data2/zhangya/webqa/imgs.lineidx")
# image_batch = batch_image_id_to_images([30016255, 30112308, 30103103, 30276954], lineidx, "/nfs/data2/zhangya/webqa/imgs.tsv")
# Function to convert a batch of image IDs to actual images
def batch_image_id_to_images(image_ids, lineidx, file_path):
    """
    Convert a batch of image IDs to actual images.

    Args:
    image_ids (list of int): The IDs of the images.
    lineidx (list): List of line indices for image file pointers.
    file_path (str): Path to the image file.

    Returns:
    list of Image: The images corresponding to the given image IDs.
    """
    images = []
    try:
        with open(file_path, "r") as fp:
            for image_id in image_ids:
                fp.seek(lineidx[int(image_id) % 10000000])
                _, img_base64 = fp.readline().strip().split('\t')
                image = Image.open(BytesIO(base64.b64decode(img_base64)))
                images.append(image)
    except Exception as e:
        print(f"An error occurred: {e}")
    return images


def fetch_images_by_id(image_ids, lineidx_file, tsv_file):
    lineidx = load_line_indices(lineidx_file)
    return batch_image_id_to_images(image_ids, lineidx, tsv_file)
####################################################################################

from sentence_transformers import util
from models import InstructBlip
from transformers import CLIPModel, CLIPProcessor
import torch

from .retrieval import RetrievalResult

class ClipRetrieval():
    def __init__(self, example, hyparams):
        # Extracting parameters from hyparams
        self.mode = hyparams.get('mode', 'img')  # Default to multimodal
        device = hyparams.get('device', 'cuda')
        checkpoint = hyparams.get('clip_checkpoint', None) #Assuming checkpoint is mandatory
        generation_config = hyparams.get('generation_config', {})
        bnb_config = hyparams.get('bnb_config', {})
        lineidx_file = hyparams.get('lineidx_file', '/nfs/data2/zhangya/webqa/imgs.lineidx')
        tsv_file = hyparams.get('tsv_file', '/nfs/data2/zhangya/webqa/imgs.tsv')


        self.device = device
        #self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = example['clip_processor']
        self.model = example['clip_model']
        self.instruct_blip = example['instruct_blip']
        # self.model.to(self.device)
        from unittest.mock import MagicMock

        # Create a mock InstructBlip object
        # self.instruct_blip = MagicMock()
        # self.instruct_blip.generate_caption.return_value = "A mock caption for the image."
        # self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        # self.instruct_blip = InstructBlip(
        #     checkpoint="Salesforce/instructblip-vicuna-7b", 
        #     device=self.device, 
        #     generation_config=, bnb_config
        # )

        # device = "cpu"

        # self.instruct_blip = InstructBlip(
        #     checkpoint="Salesforce/instructblip-vicuna-7b", 
        #     device=device, 
        #     generation_config={
        #     "max_length": 128,
        #     "num_beams": 5,
        #     "do_sample": False,
        #     #"temperature": 0.7
        # },
        #     bnb_config= {
        #     "load_in_4bit": False,
        #     "load_in_8bit": True
        # })

        self.embeddings = {}  # Dictionary to store embeddings
        self.source_material = {}  # Dictionary to store source material

        self.example = example
        self.lineidx_file = lineidx_file
        self.tsv_file = tsv_file
        self.load_data()

    # def load_data(self):
    #     path = self.example['path']
    #     guid = self.example['Guid']
    #     self.db = json.loads(Path(path).read_text())[guid]

    #     for fact_type in ['txt_posFacts', 'txt_negFacts', 'img_posFacts', 'img_negFacts']:
    #         if 'txt' in fact_type:
    #             for item in self.db.get(fact_type, []):
    #                 text = item['fact']
    #                 id = item['snippet_id']  # Assuming each item has a unique snippet_id
    #                 self.add_to_index(id, text=text)
    #         elif 'img' in fact_type:
    #             image_ids = [item['image_id'] for item in self.db.get(fact_type, [])]
    #             images = fetch_images_by_id(image_ids, self.lineidx_file, self.tsv_file)
    #             for image_id, image in zip(image_ids, images):
    #                 self.add_to_index(image_id, image_path=image)

    def load_data(self):
        path = self.example['path']
        guid = self.example['Guid']
        self.db = json.loads(Path(path).read_text())[guid]

        if self.mode in ['mm', 'txt']:
            self.load_text_data()
        if self.mode in ['mm', 'img']:
            self.load_image_data()

    def load_text_data(self):
        for fact_type in ['txt_posFacts', 'txt_negFacts']:
            for item in self.db.get(fact_type, []):
                text = item['fact']
                id = item['snippet_id']  # Assuming each item has a unique snippet_id
                self.add_to_index(id, text=text)
            
    def load_image_data(self):
        for fact_type in ['img_posFacts', 'img_negFacts']:
            image_ids = [item['image_id'] for item in self.db.get(fact_type, [])]
            images = fetch_images_by_id(image_ids, self.lineidx_file, self.tsv_file)
            for image_id, image in zip(image_ids, images):
                self.add_to_index(image_id, image_path=image)

    def add_to_index(self, id, text=None, image_path=None):
        if text:
            text_emb = self.embed_text(text)
            self.embeddings[id] = {'type': 'text', 'embedding': text_emb}
        if image_path:
            image_emb = self.embed_image(image_path)
            self.embeddings[id] = {'type': 'image', 'embedding': image_emb}
        
        if text:
            self.source_material[id] = {'type': 'text', 'content': text}
        if image_path:
            self.source_material[id] = {'type': 'image', 'content': image_path}

    def embed_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def embed_image(self, image_path):
        image = self.processor(images=image_path, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**image)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def search(self, query, top_k=5):
        query_emb = self.embed_text(query) # [1,512]
        #corpus_embeddings = [self.embeddings[key]['embedding'] for key in self.embeddings]
        corpus_embeddings = torch.stack([self.embeddings[key]['embedding'].squeeze() for key in self.embeddings]) # [34,512]
        search_results = util.semantic_search(query_emb, corpus_embeddings, top_k=top_k)[0]
        return search_results

    def retrieve(self, state, query):
        # search_results = self.search(query)
        # retrieved_snippets = []
        # retrieved_sources = []
        # flags = []
        # relevance_scores = []

        # for result in search_results:
        #     corpus_id = result['corpus_id']
        #     score = result['score']
        #     id = list(self.embeddings.keys())[corpus_id]
        #     emb_data = self.embeddings[id]

        #     retrieved_sources.append(id)
        #     relevance_scores.append(score)

        #     if emb_data['type'] == 'text':
        #         retrieved_snippets.append(emb_data['text'])
        #         flags.append('text')
        #     elif emb_data['type'] == 'image':
        #         image_path = emb_data['image_path']
        #         caption = self.instruct_blip.generate_caption("", image_path)  # Assuming empty text prompt
        #         retrieved_snippets.append(caption)
        #         flags.append('image')

        search_results = self.search(query)
        retrieved_snippets = []
        retrieved_sources = []
        flags = []
        relevance_scores = []

        for result in search_results:
            corpus_id = result['corpus_id']
            score = result['score']
            id = list(self.source_material.keys())[corpus_id]
            source_data = self.source_material[id]

            retrieved_sources.append(id)
            relevance_scores.append(score)

            if self.mode == 'txt':
                # Handle text-only retrieval
                retrieved_snippets.append(source_data['content'])
                flags.append('text')
            elif self.mode == 'img':
                # Handle image-only retrieval
                # For images, you might want to generate captions or just return the image paths
                image_path = source_data['content']
                caption = self.instruct_blip.generate_caption(self.example['Q'], image_path)  # Assuming empty text prompt
                retrieved_snippets.append(caption)
                flags.append('image')
            else:
                if source_data['type'] == 'text':
                    retrieved_snippets.append(source_data['content'])
                    flags.append('text')
                elif source_data['type'] == 'image':
                    image_path = source_data['content']
                    caption = self.instruct_blip.generate_caption(self.example['Q'], image_path)  # Assuming empty text prompt
                    retrieved_snippets.append(caption)
                    flags.append('image')

        result = RetrievalResult(
            state_type="RETRIEVE",
            context=query,
            retrieved_snippets=retrieved_snippets,
            retrieved_sources=retrieved_sources,
            flags=flags,
            relevance_scores=relevance_scores
        )
        return result
    
    def generate_questions(self, query):
        search_results = self.search(query, top_k=5)
        questions = []
        for id, _ in search_results:
            if self.embeddings[id]['type'] == 'image':
                question = self.generate_question_from_image(self.embeddings[id]['path'])
                questions.append(question)
        return questions

    def generate_question_from_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.blip_model.generate(**inputs)
        question = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        return question

# Example Usage
# retrieval = ClipRetrieval()
# retrieval.add_to_index('id1', text='A sample text')
# retrieval.add_to_index('id2', image_path='path_to_image.jpg')
# results = retrieval.search('query text')
# questions = retrieval.generate_questions('query text')

if __name__ == "__main__":
    # Example usage
    example = {
        'path': '/home/stud/abinder/master-thesis/data/n_samples_50_split_val_solution_img_seed_42_1691423195.1279488_samples_dict.json',
        'Guid': 'd5c4710c0dba11ecb1e81171463288e9'
    }
    retrieval = ClipRetrieval(example, {'mode': 'img'})
    result = retrieval.retrieve("\"What's larger: Amanita muscaria or Schizophyllum commune?\"")