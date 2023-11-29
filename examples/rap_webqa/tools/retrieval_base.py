class BaseRetrieval:
    def __init__(self):
        self.embeddings = {}
        self.source_material = {}

    def load_data(self):
        # Implementation for loading data
        pass

    def add_to_index(self, id, content, is_gold=False):
        # General implementation for adding to index
        pass

    def search(self, query, top_k):
        # Template method for performing a search
        pass

    def retrieve(self, state, query):
        # Template method for processing retrieval results
        pass

    def process_search_results(self, search_results, query):
        # General method for processing search results
        pass