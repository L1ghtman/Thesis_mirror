import time
from components import helpers, custom_llm, custom_sim_eval

from gptcache import cache, Config
from gptcache.manager import manager_factory
from gptcache.embedding import Onnx
from gptcache.processor.post import temperature_softmax
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.processor.pre import get_prompt

onnx = Onnx()
data_manager = manager_factory("sqlite,faiss", vector_params={"dimension": onnx.dimension})

cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    post_process_messages_func=temperature_softmax
    )
# cache.config = Config(similarity_threshold=0.2)

llm = custom_llm.localLlama()

questions = [
    "what's github",
    "what is github",
]

cached_llm = LangChainLLMs(llm=llm)

for question in questions:
        start_time = time.time()
        answer = cached_llm(prompt=question, cache_obj=cache)
        print(f"Question: {question}")
        print("Time consuming: {:.2f}s".format(time.time() - start_time))
        print(f"Answer: {answer}\n")
        # print this line in blue color
        print("\033[94m" + "-----------------------------------------------------------" + "\033[0m\n")