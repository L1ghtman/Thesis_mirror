from sentence_transformers import SentenceTransformer

def embed_sentence(sentence, model):
    """Embed a sentence using the SentenceTransformer model."""
    return model.encode(sentence)
