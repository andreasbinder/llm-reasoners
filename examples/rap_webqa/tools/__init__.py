from .retrieval_base import MPNetRetrieval, ClipRetrieval, RetrievalResult, Query

def get_retriever(model_name):
    if model_name == 'mpnet':
        return MPNetRetrieval
    elif model_name == 'clip':
        return ClipRetrieval
    else:
        raise ValueError("Unknown retrieval model")