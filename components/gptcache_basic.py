import time
import helpers, custom_llm

from gptcache.adapter.api import init_similar_cache
from gptcache.processor.pre import last_content

init_similar_cache(pre_func=last_content)

llm = custom_llm.localLlama()

questions = [
    "Please expalin what github is. For testing purposes, end your response with the word 'end'.",
    #"can you explain what GitHub is",
    #"can you tell me more about GitHub",
    #"what is the purpose of GitHub",
]

for question in questions:
    start_time = time.time()
    response = llm.invoke(question)
    print(f"Question: {question}")
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {response}\n')