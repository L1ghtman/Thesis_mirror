from components.mini_batch_kmeans import MiniBatchKMeansClustering
from components.custom_llm import custom_llm
from gptcache.core import Cache
from random import random
from gptcache.adapter.langchain_models import LangChainLLMs
from components.new_cache_logger import new_cache_logger
from numpy import np
import time
from components.helpers import convert_gptcache_report

class EmbeddingInterceptor:
    """
    Intercept embeddings as they are generated for clustering.
    """
    def __init__(self, original_embedding_func):
        self.original_func = original_embedding_func
        self.last_embedding = None
        self.last_query = None

    def __call__(self, prompt, **kwargs):
        self.last_query = prompt
        embedding = self.original_func(prompt, **kwargs)
        self.last_embedding = embedding
        return embedding

class SmartCache:
    def __init__(self, base_temperature=0.5, num_clusters=8):
        self.cache = Cache()
        self.clusterer = MiniBatchKMeansClustering(num_clusters=num_clusters)
        self.base_temperature = base_temperature
        self.min_temperature = 0.0
        self.max_temperature = 2.0
        self.llm = custom_llm.localLlama()
        self.cache_llm = LangChainLLMs(llm=self.llm)
        #self.CachLogger = new_cache_logger.CacheLogger()

    def init(self, 
             embedding_func, 
             data_manager, 
             similarity_evaluation, 
             pre_embedding_func
             ):
        """
        Initialize cache with embedding function, data manager, sim eval, and pre-embedding function.
        """
        self.cache.init(
            pre_embedding_func=pre_embedding_func,
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=similarity_evaluation,
        )

    def get_effective_temperature(self, query_embedding):
        """
        Calculate the temperature to be used based on:
        1. base temperature
        2. cache population adjustment
        3. cluste based adjustment
        """

        temp = self.base_temperature

        cache_size = self.cache.data_manager.count()
        cache_adjustment = 1.0 / max(1, np.log2(cache_size + 1))

        cluster_adjustment = self.clusterer.get_temperature_adjustment(query_embedding)

        effective_temperature = temp * (cache_adjustment * cluster_adjustment) / 2

        return max(self.min_temperature, min(self.max_temperature, effective_temperature))
    
    def process_query(self, query, embedding, semantic_cache, CachLogger):
        """
        Process query with dynamic temperature adjustment.

        Args:
            query: the query string
            embedding: the embedding of the query
            semantic_cache: the cache object
            CachLogger: the logger object for logging requests
        """
        # Get the effective temperature based on the query embedding 
        temp = self.get_effective_temperature(embedding)

        self.clusterer.add_to_buffer(embedding)
        self.clusterer.process_buffer()

        # Decide whether to use cache based on the effective temperature
        use_cache = random.random() < temp / self.max_temperature

        # Set up stats for logging
        pre_stats = {
            "hits": semantic_cache.report.hint_cache_count,
            "llm_calls": semantic_cache.report.op_llm.count,
        }

        # Start the timer
        start_time = time.time()

        # Use cache or call LLM directly
        if use_cache:
            answer = self.cache_llm(prompt=query, cache_obj=semantic_cache)
            is_hit = semantic_cache.report.hint_cache_count > pre_stats["hits"]
        else:
            answer = self.llm.invoke(query)
            is_hit = False
            semantic_cache.import_data(
                questions=[query],
                answers=[answer],
            )
        
        response_time = time.time() - start_time

        # Log the request
        CachLogger.log_request(
            query=query,
            response=answer,
            response_time=response_time,
            is_cache_hit=is_hit,
            similarity_score=None,
            used_cache=use_cache,
            temperature=temp,
            report_metrics=convert_gptcache_report(semantic_cache)
        ) 
        
        print(f"Question: {query}")
        print("Time consuming: {:.2f}s".format(response_time))
        print(f"Answer: {answer}\n")
        print("\033[94m" + "-----------------------------------------------------------" + "\033[0m\n")

