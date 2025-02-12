from sentence_transformers import SentenceTransformer

def get_embeddings():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = [ 
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]

    questions = [
        "What is github? Explain briefly.",
        "can you explain what GitHub is? Explain briefly.",
        "can you tell me more about GitHub? Explain briefly.",
        "what is the purpose of GitHub? Explain briefly.",
    ]

    sentence_embeddings = model.encode(questions)

    similarities = model.similarity(sentence_embeddings, sentence_embeddings)
    print("Similarities:")
    print(similarities.shape)
    print(similarities)

def embedding():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def main():
    get_embeddings()

if __name__ == "__main__":
    main()