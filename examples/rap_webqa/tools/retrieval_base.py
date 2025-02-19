import json
from pathlib import Path
from PIL import Image
import torch
import base64
from io import BytesIO
from typing import NamedTuple, List
from unittest.mock import MagicMock

import base64
from PIL import Image
from io import BytesIO

from sentence_transformers.cross_encoder import CrossEncoder

# Function to load line indices
def load_line_indices(file_path):
    with open(file_path, "r") as fp_lineidx:
        return [int(i.strip()) for i in fp_lineidx.readlines()]


# Usage Example
# from utils.webqa_helpers import load_line_indices, batch_image_id_to_images
# lineidx = load_line_indices("/nfs/data2/zhangya/webqa/imgs.lineidx")
# image_batch = batch_image_id_to_images([30016255, 30112308, 30103103, 30276954], lineidx, "/nfs/data2/zhangya/webqa/imgs.tsv")
# Function to convert a batch of image IDs to actual images
def batch_image_id_to_images(image_ids, lineidx, file_path, raw=False):
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
                #image = Image.open(BytesIO(base64.b64decode(img_base64)))
                if raw:
                    image = base64.b64decode(img_base64).convert("RGB")
                else:
                    image = Image.open(BytesIO(base64.b64decode(img_base64))).convert("RGB")
                    
                images.append(image)
    except Exception as e:
        print(f"An error occurred: {e}")
    return images


def fetch_images_by_id(image_ids, lineidx_file, tsv_file, raw=False):
    lineidx = load_line_indices(lineidx_file)
    return batch_image_id_to_images(image_ids, lineidx, tsv_file, raw)

class RetrievalBase():
    def __init__(self, example, hyparams):

        # Common attributes from 'example'
        self.example = example
        self.embedding_model = example.get('embedding_model')
        self.caption_model = example.get('caption_model', '')
        self.guid = example.get('Guid')  # Assuming 'Guid' is common

        # Common attributes from 'hyparams'
        self.mode = hyparams.get('mode', 'mm')  # Default to 'img'
        self.device = hyparams.get('device', 'cuda')
        self.top_k = hyparams.get('top_k', 8)
        self.adjust_mod_bias = hyparams.get('adjust_mod_bias', False)
        # self.lineidx_file = hyparams.get('lineidx_file', '/default/path/to/imgs.lineidx')
        # self.tsv_file = hyparams.get('tsv_file', '/default/path/to/imgs.tsv')
        self.lineidx_file = hyparams.get('lineidx_file', '/nfs/data2/zhangya/webqa/imgs.lineidx')
        self.tsv_file = hyparams.get('tsv_file', '/nfs/data2/zhangya/webqa/imgs.tsv')
        self.path_to_para = hyparams.get(
            'path_to_para', 
            '/home/stud/abinder/master-thesis/data/n_samples_50_split_val_solution_img_seed_42_1691423195.1279488_samples_dict_paraphrased.json'
            )
        self.use_caption_model = hyparams.get('use_caption_model', False)

        self.caption_model_prompt = hyparams['caption_model'].get('prompt', '{query}')
        self.caption_model_max_new_tokens = hyparams['caption_model'].get('max_new_tokens', 512)
        self.caption_model_filter_token = hyparams['caption_model'].get('filter_token', 'IGNORE')

        # 
        self.use_captions = hyparams.get('use_captions', False)

        # Initialize embeddings and source material dictionaries
        self.embeddings = {}
        self.source_material = {}

        # Load data if necessary
        self.load_data()        
        
    # TODO shared
    def load_data(self):
        # path = self.example['path']
        # guid = self.example['Guid']
        # self.db = json.loads(Path(path).read_text())[guid]

        # if self.mode in ['mm', 'txt']:
        #     self.load_text_data()
        # if self.mode in ['mm', 'img']:
        #     self.load_image_data()

        pass

    # TODO shared
    def load_text_data(self):
        # Load and index positive text facts
        for item in self.db.get('txt_posFacts', []):
            self.add_to_index(item['snippet_id'], text=item['fact'], is_gold=True)

        # Load and index negative text facts
        for item in self.db.get('txt_negFacts', []):
            self.add_to_index(item['snippet_id'], text=item['fact'], is_gold=False)

        # Load and index negative text facts
        for item in self.db.get('txt_Facts', []):
            self.add_to_index(item['snippet_id'], text=item['fact'], is_gold=False)

    def load_image_data(self):
        pass

    def add_to_index(self, id, text=None, image_path=None, is_gold=False):
        pass

    def normalize_scores(self,scores):
        return (scores + 1) / 2
    
    def adjust_scores_by_modality(self, scores, adjust_mod_bias):
        if not adjust_mod_bias:
            return scores

        # Initialize lists to store scores by modality
        image_scores = []
        text_scores = []

        # Categorize scores by modality
        for idx, (id, embedding) in enumerate(self.embeddings.items()):
            if embedding['type'] == 'image':
                image_scores.append(scores[idx])
            elif embedding['type'] == 'text':
                text_scores.append(scores[idx])

        # Convert lists to tensors
        image_scores = torch.tensor(image_scores)
        text_scores = torch.tensor(text_scores)

        # Calculate mean scores and adjust
        mean_image_score = torch.mean(image_scores)
        mean_text_score = torch.mean(text_scores)

        score_diff = abs(mean_image_score - mean_text_score)
        if mean_image_score < mean_text_score:
            for idx, (id, embedding) in enumerate(self.embeddings.items()):
                if embedding['type'] == 'image':
                    scores[idx] += score_diff
        else:
            for idx, (id, embedding) in enumerate(self.embeddings.items()):
                if embedding['type'] == 'text':
                    scores[idx] += score_diff

        return scores

    def get_num_sources(self):
        return len(self.source_material)

    def search(self, query, top_k=None, adjust_mod_bias=False):
        if top_k is None:
            top_k = self.top_k

        # Ensure top_k does not exceed the number of items in the corpus
        corpus_size = len(self.embeddings)
        top_k = min(top_k, corpus_size)


        query_emb = self.embedding_model.embed_text(query) # [1,512]
        corpus_embeddings = torch.stack([self.embeddings[key]['embedding'].squeeze() for key in self.embeddings]) # [N,512]

        # Compute dot products (scores) between query and corpus embeddings
        scores = torch.matmul(corpus_embeddings, query_emb.T).squeeze()

        if self.adjust_mod_bias:
            scores = self.adjust_scores_by_modality(scores, self.adjust_mod_bias)

        scores = self.normalize_scores(scores)

        # Get top-k results based on scores
        top_results_indices = torch.topk(scores, top_k).indices
        search_results = [{'corpus_id': idx.item(), 'score': scores[idx].item()} for idx in top_results_indices]

        return search_results

    def retrieve(self, state, query):
        
        search_results = self.search(query)
        retrieved_data = self.process_search_results(search_results, query)

        print("#" * 25 + "RETRIEVE" + "#" * 25)
        print(query)
        for item1, item2 in zip(retrieved_data['modalities'], retrieved_data['snippets']):

            print(item1, item2, sep='\n')
            print("#" * 25)

        result = RetrievalResult(
            state_type="RETRIEVE",
            context=query,
            retrieved_snippets=retrieved_data['snippets'],
            retrieved_sources=retrieved_data['sources'],
            modalities=retrieved_data['modalities'],
            relevance_scores=retrieved_data['scores'],
            is_gold=retrieved_data['is_gold'],
            is_filtered=retrieved_data['is_filtered']
        )
        return result


    # def get_caption_model_prompt(self, query):
    #     overall_question = self.example['Q']
    #     return self.caption_model_prompt.format(query=query, overall_question=overall_question)

    def get_caption_model_prompt(self, query, caption):
        
        return self.caption_model_prompt.format(
            query=query, 
            caption=caption,
            filter_token=self.caption_model_filter_token,
            overall_question=self.example['Q']
            )
    
    def sanitize_vlm_output(self, output):
        return output.replace('\n', '').strip()

    def process_search_results(self, search_results, query):
        snippets = []
        sources = []
        modalities = []
        scores = []
        is_gold = []
        is_filtered = []

        for result in search_results:
            id = list(self.source_material.keys())[result['corpus_id']]
            source_data = self.source_material[id]

            content = source_data['content']
            content_type = source_data['type']

            # Check if the content type is 'image'
            if content_type == 'image' and self.use_caption_model:
                # Fetch the image using its ID
                image = fetch_images_by_id(
                    [id],
                    self.lineidx_file,
                    self.tsv_file
                    )
                # Generate a caption for the image based on the query

                


                original_caption = source_data['caption']
                caption_model_prompt = self.get_caption_model_prompt(query, original_caption)

                print("#" * 25 + "Start" + str(id) + "#" * 25)
                print
                print(caption_model_prompt)
                print("#" * 25 )

                caption = self.generate_image_caption(caption_model_prompt, image)
                caption = self.sanitize_vlm_output(caption)
                #caption = self.generate_image_caption(query, image)
                print(caption)
                print("#" * 25 )

                content = caption

                #if content == 'NONE':
                if self.caption_model_filter_token in content:
                    is_filtered.append(True)
                else:
                    is_filtered.append(False)
            
            else:
                is_filtered.append(False)

            sources.append(id)
            scores.append(result['score'])
            snippets.append(content)
            modalities.append(content_type)
            is_gold.append(source_data['is_gold'])

        return {'snippets': snippets, 'sources': sources, 'modalities': modalities, 'scores': scores, 'is_gold': is_gold, "is_filtered": is_filtered}

    def process_batch_search_results(self, search_results, query):
        # Separate containers for text and image results
        text_snippets = []
        image_ids = []
        image_data = []

        # Containers for final results
        snippets = []
        sources = []
        modalities = []
        scores = []
        is_gold = []

        # First pass to separate text and image results
        for result in search_results:
            id = list(self.source_material.keys())[result['corpus_id']]
            source_data = self.source_material[id]

            content_type = source_data['type']
            if content_type == 'image' and self.use_caption_model:
                image_ids.append(id)
                image_data.append(source_data['content'])  # Assuming this is the image data
            else:
                text_snippets.append(source_data['content'])

        # Batch process all images
        if image_ids:
            images = fetch_images_by_id(image_ids, self.lineidx_file, self.tsv_file)
            caption_model_prompt = self.get_caption_model_prompt(query)
            #generate_batch(self, prompts, images)
            captions = self.generate_batch(caption_model_prompt, images)
            captions = [self.sanitize_vlm_output(caption) for caption in captions]

        # Reassemble results with text and image content
        image_index = 0
        for result in search_results:
            id = list(self.source_material.keys())[result['corpus_id']]
            source_data = self.source_material[id]

            content_type = source_data['type']
            if content_type == 'image' and self.use_caption_model:
                content = captions[image_index]
                image_index += 1
            else:
                content = text_snippets.pop(0)

            sources.append(id)
            scores.append(result['score'])
            snippets.append(content)
            modalities.append(content_type)
            is_gold.append(source_data['is_gold'])

        return {'snippets': snippets, 'sources': sources, 'modalities': modalities, 'scores': scores, 'is_gold': is_gold}


    def get_snippet_and_modality(self, source_data, query):
        if source_data['type'] == 'text':
            return source_data['content'], 'text'
        elif source_data['type'] == 'image':
            image_path = source_data['content']
            caption = self.generate_image_caption(query, image_path)
            return caption, 'image'

    def generate_image_caption(self, query, image_path):
        # Placeholder for image caption generation logic
        
        return self.caption_model.generate_caption(prompt=query, 
                                                   image_file=image_path, 
                                                   max_new_tokens=self.caption_model_max_new_tokens)  # Assuming empty text prompt

class ClipRetrieval(RetrievalBase):
    def __init__(self, example, hyparams):
        #super().__init__() # TODO think if parent necessar
        super().__init__(example, hyparams)

    def load_data(self):
        path = self.example['path']
        guid = self.example['Guid']
        self.db = json.loads(Path(path).read_text())[guid]

        if self.mode in ['mm', 'txt']:
            self.load_text_data()
        if self.mode in ['mm', 'img']:
            self.load_image_data()


    def load_image_data(self):
        # Fetch and index positive image facts
        pos_image_ids, captions = zip(**[(item['image_id'], item['caption']) for item in self.db.get('img_posFacts', [])])
        pos_images = fetch_images_by_id(pos_image_ids, self.lineidx_file, self.tsv_file)
        for image_id, image, caption in zip(pos_image_ids, pos_images, captions):
            self.add_to_index(
                image_id, 
                image_path=image,
                caption=caption,
                is_gold=True)

        # Fetch and index negative image facts
        neg_image_ids, captions = zip(**[(item['image_id'], item['caption']) for item in self.db.get('img_negFacts', [])])
        neg_images = fetch_images_by_id(neg_image_ids, self.lineidx_file, self.tsv_file)
        for image_id, image, caption in zip(neg_image_ids, neg_images, captions):
            self.add_to_index(image_id, image_path=image, caption=caption, is_gold=False)

        # Fetch and index negative image facts
        neg_image_ids, captions = zip([(item['image_id'], item['caption']) for item in self.db.get('img_Facts', [])])
        neg_images = fetch_images_by_id(neg_image_ids, self.lineidx_file, self.tsv_file)
        for image_id, image, caption in zip(neg_image_ids, neg_images, captions):
            self.add_to_index(image_id, image_path=image, caption=caption, is_gold=False)

    def add_to_index(self, id, text=None, image_path=None, is_gold=False, caption=None):
        if text:
            #text_emb = self.embed_text(text)
            text_emb = self.embedding_model.embed_text(text)
            self.embeddings[id] = {'type': 'text', 'embedding': text_emb}
            self.source_material[id] = {'type': 'text', 'content': text, 'is_gold': is_gold}
        if image_path:
            #image_emb = self.embed_image(image_path)
            image_emb = self.embedding_model.embed_image(image_path)
            self.embeddings[id] = {'type': 'image', 'embedding': image_emb}
            self.source_material[id] = {'type': 'image', 'content': image_path, 'is_gold': is_gold, 'caption': caption}

class MPNetRetrieval(RetrievalBase):
    def __init__(self, example, hyparams):
        #super().__init__() # TODO think if parent necessar
        super().__init__(example, hyparams)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')   

    def load_data(self):
        path = self.example['path']
        guid = self.example['Guid']
        self.db = json.loads(Path(path).read_text())[guid]

        self.db_para = json.loads(Path(self.path_to_para).read_text())[self.guid]

        if self.mode in ['mm', 'txt']:
            self.load_text_data()
        if self.mode in ['mm', 'img']:
            self.load_image_data()


    def load_image_data(self):
        # Fetch and index positive image facts
        for item in self.db.get('img_posFacts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                caption=item['caption'], 
                is_gold=True
                )
            
        for item in self.db.get('img_negFacts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                caption=item['caption'], 
                is_gold=False
                )
            
        for item in self.db.get('img_Facts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                caption=item['caption'], 
                is_gold=False
                )

    def add_to_index(self, id, text=None, image_path=None, is_gold=False, caption=None):
        if text:
            #text_emb = self.embed_text(text)
            text_emb = self.embedding_model.embed_text(text)
            self.embeddings[id] = {'type': 'text', 'embedding': text_emb}
            self.source_material[id] = {'type': 'text', 'content': text, 'is_gold': is_gold}
        if image_path:
            #image_emb = self.embed_image(image_path)
            #image_emb = self.embedding_model.embed_image(image_path)
            if self.use_captions:
                image_path = caption + ' | ' + image_path 
                image_path = image_path.strip()
            image_emb = self.embedding_model.embed_text(image_path)
            self.embeddings[id] = {'type': 'image', 'embedding': image_emb}
            self.source_material[id] = {'type': 'image', 'content': image_path, 'is_gold': is_gold, 'caption': caption}

    # def search(self, query, top_k=None, adjust_mod_bias=False):
    #     if top_k is None:
    #         top_k = self.top_k

    #     # Ensure top_k does not exceed the number of items in the corpus
    #     corpus_size = len(self.embeddings)
    #     top_k = min(top_k, corpus_size)


    #     query_emb = self.embedding_model.embed_text(query) # [1,512]
    #     corpus_embeddings = torch.stack([self.embeddings[key]['embedding'].squeeze() for key in self.embeddings]) # [N,512]

    #     # Compute dot products (scores) between query and corpus embeddings
    #     scores = torch.matmul(corpus_embeddings, query_emb.T).squeeze()

    #     if self.adjust_mod_bias:
    #         scores = self.adjust_scores_by_modality(scores, self.adjust_mod_bias)

    #     scores = self.normalize_scores(scores)

    #     # Get top-k results based on scores
    #     top_results_indices = torch.topk(scores, top_k).indices

    #     guids = list(self.source_material.keys())
    #     input_texts = [
    #         [query, self.source_material[guids[idx.item()]]['content']] for idx in top_results_indices
    #     ]
    #     #input_texts = [query] + [self.source_material[key]['content'] for key in self.source_material]

    #     cross_scores = self.cross_encoder.predict(input_texts, apply_softmax=True)

    #     import numpy as np

        

    #     rerank = 4
    #     # Sort the scores in decreasing order
    #     sim_scores_argsort = list(reversed(np.argsort(cross_scores)))[:rerank]

    #     print("Scores")
    #     print(scores)
    #     print("Cross Scores")
    #     print(cross_scores)
    #     print("Sim Scores Argsort")
    #     print(sim_scores_argsort)


    #     final_results_indices = [top_results_indices[idx].item() for idx in sim_scores_argsort]

    #     search_results = [{'corpus_id': idx, 'score': scores[idx]} for idx in final_results_indices]

    #     return search_results



class Query(MPNetRetrieval):

    # def __init__(self, example, hyparams):
    #     #super().__init__() # TODO think if parent necessar
    #     super().__init__(example, hyparams)

    def __init__(self, example, hyparams):
        #super().__init__() # TODO think if parent necessar
        super().__init__(example, hyparams)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')   

    def retrieve(self, state, query):
        
        query = self.example['Q']
        search_results = self.search(query)
        retrieved_data = self.process_search_results(search_results, query)

        print("#" * 25 + "RETRIEVE" + "#" * 25)
        print(query)
        for item1, item2 in zip(retrieved_data['modalities'], retrieved_data['snippets']):

            print(item1, item2, sep='\n')
            print("#" * 25)

        result = RetrievalResult(
            state_type="RETRIEVE",
            context=query,
            retrieved_snippets=retrieved_data['snippets'],
            retrieved_sources=retrieved_data['sources'],
            modalities=retrieved_data['modalities'],
            relevance_scores=retrieved_data['scores'],
            is_gold=retrieved_data['is_gold'],
            is_filtered=retrieved_data['is_filtered']
        )
        return result

    def search(self, query, top_k=None, adjust_mod_bias=False):
        if top_k is None:
            top_k = self.top_k

        # Ensure top_k does not exceed the number of items in the corpus
        corpus_size = len(self.embeddings)
        top_k = min(top_k, corpus_size)


        query_emb = self.embedding_model.embed_text(query) # [1,512]
        corpus_embeddings = torch.stack([self.embeddings[key]['embedding'].squeeze() for key in self.embeddings]) # [N,512]

        # Compute dot products (scores) between query and corpus embeddings
        scores = torch.matmul(corpus_embeddings, query_emb.T).squeeze()

        if self.adjust_mod_bias:
            scores = self.adjust_scores_by_modality(scores, self.adjust_mod_bias)

        scores = self.normalize_scores(scores)

        # Get top-k results based on scores
        top_results_indices = torch.topk(scores, top_k).indices

        guids = list(self.source_material.keys())
        input_texts = [
            [query, self.source_material[guids[idx.item()]]['content']] for idx in top_results_indices
        ]
        #input_texts = [query] + [self.source_material[key]['content'] for key in self.source_material]

        cross_scores = self.cross_encoder.predict(input_texts, apply_softmax=True)

        import numpy as np

        

        rerank = 4
        # Sort the scores in decreasing order
        sim_scores_argsort = list(reversed(np.argsort(cross_scores)))[:rerank]

        print("Scores")
        print(scores)
        print("Cross Scores")
        print(cross_scores)
        print("Sim Scores Argsort")
        print(sim_scores_argsort)


        final_results_indices = [top_results_indices[idx].item() for idx in sim_scores_argsort]

        search_results = [{'corpus_id': idx, 'score': scores[idx]} for idx in final_results_indices]

        return search_results


class DPRRetrieval(RetrievalBase):
    def __init__(self, example, hyparams):
        #super().__init__() # TODO think if parent necessar
        super().__init__(example, hyparams)

    def load_data(self):
        path = self.example['path']
        guid = self.example['Guid']
        self.db = json.loads(Path(path).read_text())[guid]

        self.db_para = json.loads(Path(self.path_to_para).read_text())[self.guid]

        if self.mode in ['mm', 'txt']:
            self.load_text_data()
        if self.mode in ['mm', 'img']:
            self.load_image_data()


    def load_image_data(self):
        # Fetch and index positive image facts
        for item in self.db.get('img_posFacts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                is_gold=True
                )
            
        for item in self.db.get('img_negFacts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                is_gold=False
                )
            
        for item in self.db.get('img_Facts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                is_gold=False
                )

    def add_to_index(self, id, text=None, image_path=None, is_gold=False):
        if text:
            #text_emb = self.embed_text(text)
            text_emb = self.embedding_model.embed_text(text)
            self.embeddings[id] = {'type': 'text', 'embedding': text_emb}
            self.source_material[id] = {'type': 'text', 'content': text, 'is_gold': is_gold}
        if image_path:
            #image_emb = self.embed_image(image_path)
            #image_emb = self.embedding_model.embed_image(image_path)
            image_emb = self.embedding_model.embed_text(image_path)
            self.embeddings[id] = {'type': 'image', 'embedding': image_emb}
            self.source_material[id] = {'type': 'image', 'content': image_path, 'is_gold': is_gold}

    def search(self, query, top_k=None, adjust_mod_bias=False):
        if top_k is None:
            top_k = self.top_k

        # Ensure top_k does not exceed the number of items in the corpus
        corpus_size = len(self.embeddings)
        top_k = min(top_k, corpus_size)


        query_emb = self.embedding_model.embed_query(query) # [1,512]
        corpus_embeddings = torch.stack([self.embeddings[key]['embedding'].squeeze() for key in self.embeddings]) # [N,512]

        # Compute dot products (scores) between query and corpus embeddings
        scores = torch.matmul(corpus_embeddings, query_emb.T).squeeze()

        if self.adjust_mod_bias:
            scores = self.adjust_scores_by_modality(scores, self.adjust_mod_bias)

        scores = self.normalize_scores(scores)

        # Get top-k results based on scores
        top_results_indices = torch.topk(scores, top_k).indices
        search_results = [{'corpus_id': idx.item(), 'score': scores[idx].item()} for idx in top_results_indices]

        return search_results

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class CrossRetrieval(RetrievalBase):
    
    def __init__(self, example, hyparams):
        #super().__init__() # TODO think if parent necessar
        super().__init__(example, hyparams)
        model_name='cross-encoder/nli-deberta-v3-large'
        self.top_k=4
        #self.corpus = corpus
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.top_k = top_k


    def load_data(self):
        path = self.example['path']
        guid = self.example['Guid']
        self.db = json.loads(Path(path).read_text())[guid]

        self.db_para = json.loads(Path(self.path_to_para).read_text())[self.guid]

        if self.mode in ['mm', 'txt']:
            self.load_text_data()
        if self.mode in ['mm', 'img']:
            self.load_image_data()


    def load_image_data(self):
        # Fetch and index positive image facts
        for item in self.db.get('img_posFacts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                is_gold=True
                )
            
        for item in self.db.get('img_negFacts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                is_gold=False
                )
            
        for item in self.db.get('img_Facts', []):
            self.add_to_index(
                item['image_id'], 
                image_path=self.db_para[str(item['image_id'])], 
                is_gold=False
                )

    # def add_to_index(self, id, text=None, image_path=None, is_gold=False):
    #     if text:
    #         #text_emb = self.embed_text(text)
    #         text_emb = self.embedding_model.embed_text(text)
    #         self.embeddings[id] = {'type': 'text', 'embedding': text_emb}
    #         self.source_material[id] = {'type': 'text', 'content': text, 'is_gold': is_gold}
    #     if image_path:
    #         #image_emb = self.embed_image(image_path)
    #         #image_emb = self.embedding_model.embed_image(image_path)
    #         image_emb = self.embedding_model.embed_text(image_path)
    #         self.embeddings[id] = {'type': 'image', 'embedding': image_emb}
    #         self.source_material[id] = {'type': 'image', 'content': image_path, 'is_gold': is_gold}

    def search(self, query, top_k=None, adjust_mod_bias=False):
        # Ensure top_k does not exceed the number of items in the corpus
        #top_k = min(self.top_k, len(self.corpus))
        corpus_size = len(self.source_material)
        top_k = min(self.top_k, corpus_size)

        # Prepare model and tokenizer
        self.model.eval()

        #corpus_embeddings = torch.stack([self.embeddings[key]['embedding'].squeeze() for key in self.embeddings]) # [N,512]
        corpus = [self.source_material[key]['content'] for key in self.source_material]
        # Score each query-corpus pair
        scores = []
        for doc in corpus:
            inputs = self.tokenizer(query, doc, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
                score = torch.softmax(logits, dim=1)[:, 1]  # Assuming class 1 is the 'relevant' class
                scores.append(score.item())

        # Sort scores and get top-k results
        top_results_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        search_results = [{'corpus_id': idx, 'score': scores[idx]} for idx in top_results_indices]

        return search_results

    def add_to_index(self, id, text=None, image_path=None, is_gold=False):
        if text:
            #text_emb = self.embed_text(text)
            #text_emb = self.embedding_model.embed_text(text)
            #self.embeddings[id] = {'type': 'text', 'embedding': text_emb}
            self.source_material[id] = {'type': 'text', 'content': text, 'is_gold': is_gold}
        if image_path:
            #image_emb = self.embed_image(image_path)
            #image_emb = self.embedding_model.embed_image(image_path)
            # image_emb = self.embedding_model.embed_text(image_path)
            # self.embeddings[id] = {'type': 'image', 'embedding': image_emb}
            self.source_material[id] = {'type': 'image', 'content': image_path, 'is_gold': is_gold}

    # def search(self, query, top_k=None, adjust_mod_bias=False):
    #     if top_k is None:
    #         top_k = self.top_k

    #     # Ensure top_k does not exceed the number of items in the corpus
    #     corpus_size = len(self.embeddings)
    #     top_k = min(top_k, corpus_size)


    #     query_emb = self.embedding_model.embed_query(query) # [1,512]
    #     corpus_embeddings = torch.stack([self.embeddings[key]['embedding'].squeeze() for key in self.embeddings]) # [N,512]

    #     # Compute dot products (scores) between query and corpus embeddings
    #     scores = torch.matmul(corpus_embeddings, query_emb.T).squeeze()

    #     if self.adjust_mod_bias:
    #         scores = self.adjust_scores_by_modality(scores, self.adjust_mod_bias)

    #     scores = self.normalize_scores(scores)

    #     # Get top-k results based on scores
    #     top_results_indices = torch.topk(scores, top_k).indices
    #     search_results = [{'corpus_id': idx.item(), 'score': scores[idx].item()} for idx in top_results_indices]

    #     return search_results


import torch

#from .retrieval import RetrievalResult
#from retrieval import RetrievalResult

from typing import NamedTuple, TypedDict, Union, Tuple, List
class RetrievalResult(NamedTuple):
    state_type: str
    context: str
    retrieved_snippets: List[str]
    retrieved_sources: List[str]
    # flags: List[str] TODO removed
    modalities: List[str] # TODO new 
    relevance_scores: List[float]
    is_gold: List[bool] 
    is_filtered: List[bool] 



import json
from pathlib import Path
from PIL import Image
import torch

# TODO - Remove this hack
import sys
sys.path.append('/home/stud/abinder/master-thesis')

import base64
from PIL import Image
from io import BytesIO



import torch


#from retrieval_base import BaseRetrieval



def get_retrieval_model(model_name):
    if model_name == 'mpnet':
        return MPNetRetrieval()
    elif model_name == 'clip':
        return ClipRetrieval()
    else:
        raise ValueError("Unknown model name")

if __name__ == "__main__":

    checkpoint = 'sentence-transformers/all-mpnet-base-v2'
    #checkpoint = 'openai/clip-vit-base-patch32'
    from models.mpnet_model import MPNetEmbedder
    model = MPNetEmbedder(model_checkpoint=checkpoint,                              
                              device="cuda")
    
    # from models import ImageBind
    # model = ImageBind(pretrained=True, device='cuda'

    example = {
        'clip_processor': None, 
        'clip_model': None, 
        'instruct_blip': None,
        'path': '/home/stud/abinder/master-thesis/data/n_samples_50_split_val_solution_img_seed_42_1691423195.1279488_samples_dict.json',
        'Guid': 'd5c147b60dba11ecb1e81171463288e9', #'d5c4710c0dba11ecb1e81171463288e9',
        'Q': "\"Which building is taller, the  Old Woolworth's Department Store in Greensboro, NC, or the Warner Block building in Burlington, Vermont?\"", #"'What\'s larger: Amanita muscaria or Schizophyllum commune?'",
        'embedding_model': model,
        
        }
    hyparams = {
        'mode': 'mm', 
        'device': 'cuda',
        'top_k': 8,
        'adjust_mod_bias': True,
        'path_to_para': '/home/stud/abinder/master-thesis/data/n_samples_50_split_val_solution_img_seed_42_1691423195.1279488_samples_dict_paraphrased.json'
        }
    clip_retrieval = MPNetRetrieval(example, hyparams)

    # Mocking a search query
    mock_query = "'What\'s larger: Amanita muscaria or Schizophyllum commune?'"
    #mock_query = "A sample query"
    # top_k_results = clip_retrieval.search(mock_query, top_k=5, adjust_mod_bias=True)
    # print(top_k_results)  # Will show mocked results
    # print('#'*50)
    # Mocking a retrieval query
    #mock_query = "A sample query"
    retrieval_result = clip_retrieval.retrieve(None, mock_query)
    #print(retrieval_result)  # Will show mocked results

    from pprint import pprint
    #pprint(vars(retrieval_result)) 
    pprint({field: getattr(retrieval_result, field) for field in retrieval_result._fields}) 
    