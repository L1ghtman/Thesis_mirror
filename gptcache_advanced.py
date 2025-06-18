import os
import warnings
import time
import logging
from components import helpers, custom_llm, new_cache_logger, cache_analyzer, magnitude_based_estimator, lsh_based_estimator
import sys
import traceback
import faiss
import multiprocessing
from components.dataset_manager import DatasetManager
from gptcache.processor.pre import get_prompt
from gptcache.manager import CacheBase, VectorBase
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from gptcache.manager import get_data_manager
from gptcache.core import Cache
from components.cache_utils import embedding_func, system_cleanup, magnitude_temperature_func, lsh_temperature_func
#from components.cluster_aware_cache import ClusterAwareCache
#from components.mini_batch_kmeans import MiniBatchKMeansClustering

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CACHE_DIR = "./persistent_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

warnings.filterwarnings("ignore", message="The method `BaseLLM.__call__` was deprecated")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def process_request(question, cached_llm, semantic_cache, CacheLogger, use_cache, llm):
    """
    Send a request to the cache and log the response.
    
    Args:
        question: The question to ask
        cached_llm: The cached LLM instance
        semantic_cache: The cache object
        CacheLogger: Logger instance
    """
    pre_stats = {
        "hits": semantic_cache.report.hint_cache_count,
        "llm_calls": semantic_cache.report.op_llm.count,
    }
    
    start_time = time.time()

    # Store the last LSH debug info before the request
    last_lsh_debug = None
    if hasattr(semantic_cache, 'lsh_cache') and hasattr(semantic_cache.lsh_cache, 'estimator'):
        if hasattr(semantic_cache.lsh_cache.estimator, 'bucket_history') and semantic_cache.lsh_cache.estimator.bucket_history:
            last_bucket = semantic_cache.lsh_cache.estimator.bucket_history[-1]
            last_lsh_debug = {'last_bucket': last_bucket}

    tracking_context = {}

    answer = cached_llm(prompt=question, cache_obj=semantic_cache)
    is_hit = semantic_cache.report.hint_cache_count > pre_stats["hits"]

    temperature = None
    similarity_score = None
    lsh_debug_info = None
    hamming_distance = None

    if hasattr(semantic_cache, 'last_context') and semantic_cache.last_context:
        #cluster_id = semantic_cache.last_context.get('cluster_id')
        temperature = semantic_cache.last_context.get('temperature')
        similarity_score = semantic_cache.last_context.get('similarity_score')
        lsh_debug_info = semantic_cache.last_context.get('lsh_debug_info')

    # Calculate hamming distance if we have both current and last bucket
    if lsh_debug_info and last_lsh_debug and 'lsh_bucket' in lsh_debug_info and 'last_bucket' in last_lsh_debug:
        current_bucket = lsh_debug_info['lsh_bucket']
        last_bucket = last_lsh_debug['last_bucket']
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(current_bucket, last_bucket))

    print(f"temperature: {temperature}")
    #if lsh_debug_info:
        #print(f"LSH Debug: {lsh_debug_info}")
        #print("Got LSH debug info")

    response_time = time.time() - start_time
    report_metrics = helpers.convert_gptcache_report(semantic_cache)

    #print(f"Direct LLM calls: {semantic_cache.report.op_llm_direct.count}")
    
    CacheLogger.log_request(
        query=question,
        response=answer,
        response_time=response_time,
        is_cache_hit=is_hit,
        similarity_score=similarity_score,
        used_cache=use_cache,
        temperature=temperature,
        #magnitude=magnitude,
        #cluster_id=cluster_id,
        hamming_distance=hamming_distance,
        debug_info=lsh_debug_info,
        report_metrics=report_metrics
    )

    print(f"Question: {question}")
    print("Time consuming: {:.2f}s".format(response_time))
    print(f"Answer: {answer}\n")
    print("\033[94m" + "-----------------------------------------------------------" + "\033[0m\n")


def main():
    try:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout
        )

        evaluation = SbertCrossencoderEvaluation()
        vector_params = {
            "dimension": 384,
            "index_type": "IDMap,Flat",
            "metric_type": faiss.METRIC_L2,
            "index_path": os.path.join(CACHE_DIR, "faiss.index"),
        }
        cache_base = CacheBase("sqlite",
                               sql_url=f"sqlite:///{os.path.join(CACHE_DIR, 'cache.db')}")
        vector_base = VectorBase("faiss", **vector_params)
        data_manager = get_data_manager(cache_base, vector_base)

        # DatasetManager setup
        manager = DatasetManager()
        #manager.load_msmarco(split="train", max_samples=3)
        #manager.set_active_dataset("msmarco_train")
        #questions = manager.get_questions(dataset_name="msmarco_train")

        manager.load_from_file(
            file_path="dataset_cache/customer_qa.json",
            dataset_name="customer_qa",
            )
        manager.set_active_dataset("customer_qa")
        questions = manager.get_questions(dataset_name="customer_qa")

        partial_questions = []

        for q in questions[:1]:
            partial_questions.append(q["question"])

        test_questions = [
            "What is github? Explain briefly.",
            "can you explain what GitHub is? Explain briefly.",
            #"can you tell me more about GitHub? Explain briefly.",
            #"what is the purpose of GitHub? Explain briefly.",
            "Hello",
            #"What is the capital of US?",
            #"Tell me a communist joke",
            #"Give me a short summary of simulated annealing",
            #"What is git cherry pick",
            #"Give me a name suggestion for my dog, he likes peanut butter"
        ]

        CacheLogger = new_cache_logger.CacheLogger()
        MagnitudeCache = magnitude_based_estimator.MagnitudeCache()
        LshCache = lsh_based_estimator.LSHCache()

        llm = custom_llm.localLlama()
        cached_llm = LangChainLLMs(llm=llm)

        #clusterer = MiniBatchKMeansClustering(max_clusters=8)

        semantic_cache = Cache()
        #semantic_cache = ClusterAwareCache()
        semantic_cache.init(
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
            pre_embedding_func=get_prompt,
            #clusterer=clusterer,
            magnitude_cache=MagnitudeCache,
            lsh_cache=LshCache,
            #temperature_func=magnitude_temperature_func,
            temperature_func=lsh_temperature_func,
        )

        try:
            for question in partial_questions:
                print(f"Processing question: {question}")
                process_request(question, cached_llm, semantic_cache, CacheLogger, use_cache=True, llm=llm)
            CacheLogger.close()
            report_path = cache_analyzer.generate_latest_run_report(log_dir="cache_logs")
            print(f"Performance report saved to: {report_path}")

        finally:
           system_cleanup(semantic_cache, vector_base, data_manager)

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
