import time
from components import helpers, custom_llm

from gptcache.core import cache, Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.processor.pre import last_content, get_prompt
from gptcache.manager import get_data_manager
from gptcache.manager import CacheBase, VectorBase
from gptcache.similarity_evaluation import SearchDistanceEvaluation
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.embedding import SBERT


data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=128))

llm = custom_llm.localLlama()
sbert = SBERT()

# cache.init(
#     embedding_func=sbert,
#     data_manager=data_manager,
#     similarity_evaluation=SearchDistanceEvaluation(),
#     pre_embedding_func=get_prompt
# )

questions = [
    "What is github? Explain briefly.",
    "can you explain what GitHub is? Explain briefly.",
    "can you tell me more about GitHub? Explain briefly.",
    "what is the purpose of GitHub? Explain briefly.",
]

llm_cache = Cache()
llm_cache.init(
    pre_embedding_func=get_prompt
)

cached_llm = LangChainLLMs(llm=llm)

for question in questions:
    start_time = time.time()
    answer = cached_llm(prompt=question, cache_obj=llm_cache)
    print(f"Question: {question}")
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f"Answer: {answer}\n")
    # print this line in blue color
    print("\033[94m" + "-----------------------------------------------------------" + "\033[0m\n")
