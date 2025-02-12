import time
import logging
from components import helpers, custom_llm, custom_sim_eval

from gptcache.core import cache, Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.processor.pre import last_content, get_prompt
from gptcache.manager import CacheBase, VectorBase
from gptcache.manager import manager_factory
from gptcache.similarity_evaluation import SearchDistanceEvaluation
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.embedding import SBERT
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from gptcache.manager import get_data_manager
from gptcache.embedding import Onnx

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Embedding
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def embedding_func(prompt, extra_param=None):
    encoder = SBERT('all-MiniLM-L6-v2')
    embedding = encoder.to_embeddings(prompt)
    return tuple(embedding)

def main():
    logging.basicConfig(level=logging.DEBUG)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Evaluation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    evaluation = SbertCrossencoderEvaluation()
    #evaluation = custom_sim_eval.CustomSimilarityEvaluation()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Data Manager
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    #onnx = Onnx()

    #data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=384))
    data_manager = manager_factory("sqlite,faiss", data_dir="./workspace", vector_params={"dimension": 384})
    #data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=128))
    #data_manager = get_data_manager("data_map.txt", 1000)
    #data_manager = get_data_manager(CacheBase("mysql"), VectorBase("milvus", dimension=128), max_size=100, eviction='LRU')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# LLM
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    llm = custom_llm.localLlama()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Questions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    questions = [
        "What is github? Explain briefly.",
        #"can you explain what GitHub is? Explain briefly.",
        #"can you tell me more about GitHub? Explain briefly.",
        #"what is the purpose of GitHub? Explain briefly.",
    ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Cache initialization
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    llm_cache = Cache()
    llm_cache.init(
        pre_embedding_func=get_prompt,
        embedding_func=embedding_func,
        #embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=evaluation,
    )

    cached_llm = LangChainLLMs(llm=llm)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Run LLM with cache
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    for question in questions:
        start_time = time.time()
        answer = cached_llm(prompt=question, cache_obj=llm_cache)
        print(f"Question: {question}")
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f"Answer: {answer}\n")
        # print this line in blue color
        print("\033[94m" + "-----------------------------------------------------------" + "\033[0m\n")

if __name__ == "__main__":
    main()