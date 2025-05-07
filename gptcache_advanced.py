import os
import warnings
import time
import logging
from components import helpers, custom_llm, custom_sim_eval, new_cache_logger, cache_analyzer
import sys
import traceback
import signal
import faiss
import multiprocessing
from components.dataset_manager import DatasetManager, create_default_manager
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from gptcache import Config
from gptcache.core import Cache
from gptcache.processor.pre import get_prompt
from gptcache.manager import CacheBase, VectorBase
from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache.embedding import SBERT
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from gptcache.manager import get_data_manager
from components.cache_utils import embedding_func, system_cleanup
from components.smart_cache import SmartCache, EmbeddingInterceptor
import gptcache.adapter.adapter
from components.custom_adapter import custom_adapt
from components.mini_batch_kmeans import MiniBatchKMeansClustering


os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CACHE_DIR = "./persistent_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

warnings.filterwarnings("ignore", message="The method `BaseLLM.__call__` was deprecated")

# Configure logging to show full details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def process_questions_in_parallel(questions: List[str], cached_llm, semantic_cache, logger, max_workers: int = 8):
    """
    Process multiple questions concurrently using a thread pool.
    
    Args:
        questions: List of questions to process
        cached_llm: The cached LLM instance
        semantic_cache: The cache object
        logger: Logger instance
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of results
    """
    results = []
    
    def process_single_question(question):
        pre_stats = {
            "hits": semantic_cache.report.hint_cache_count,
            "llm_calls": semantic_cache.report.op_llm.count,
        }
        
        start_time = time.time()
        answer = cached_llm(prompt=question, cache_obj=semantic_cache)
        response_time = time.time() - start_time

        is_hit = semantic_cache.report.hint_cache_count > pre_stats["hits"]

        logger.log_request(
            query=question,
            response=answer,
            response_time=response_time,
            is_cache_hit=is_hit,
            similarity_score=None,
            used_cache=True,
            temperature=None,
            report_metrics=helpers.convert_gptcache_report(semantic_cache)
        )
        
        print(f"Question: {question}")
        print("Time consuming: {:.2f}s".format(response_time))
        print(f"Answer: {answer}\n")
        print("\033[94m" + "-----------------------------------------------------------" + "\033[0m\n")
        
        return {
            "question": question,
            "answer": answer,
            "time": response_time,
            "is_hit": is_hit
        }
    
    # Use ThreadPoolExecutor to process questions in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all questions to the executor
        future_to_question = {
            executor.submit(process_single_question, q): q 
            for q in questions
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_question):
            question = future_to_question[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Question {question} generated an exception: {exc}")
                # Might want to log this error
    
    return results

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

    # temporarily bypass cache regulation for testing
    # use_cache = True

    if use_cache:
        answer = cached_llm(prompt=question, cache_obj=semantic_cache)
        is_hit = semantic_cache.report.hint_cache_count > pre_stats["hits"]
    else:
        answer = llm.invoke(question)
        is_hit = False
        semantic_cache.import_data(
            questions=[question],
            answers=[answer],
        )

    response_time = time.time() - start_time

    CacheLogger.log_request(
        query=question,
        response=answer,
        response_time=response_time,
        is_cache_hit=is_hit,
        similarity_score=None,
        used_cache=use_cache,
        temperature=None,
        report_metrics=helpers.convert_gptcache_report(semantic_cache)
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
        interceptor = EmbeddingInterceptor(original_embedding_func=evaluation)
        #evaluation = custom_sim_eval.CustomSimilarityEvaluation()
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
        #manager = DatasetManager()
        #manager.load_msmarco(split="train", max_samples=None)
        #manager.set_active_dataset("msmarco_train")
        #questions = manager.get_questions(dataset_name="msmarco_train")

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

        test_answers = [
            "Test answer 1",
            "Test answer 2",
            "Test answer 3",
            "Test answer 4",
            "Test answer 5",
        ]

        CacheLogger = new_cache_logger.CacheLogger()

        llm = custom_llm.localLlama()
        cached_llm = LangChainLLMs(llm=llm)

        clusterer = MiniBatchKMeansClustering(num_clusters=8)

        semantic_cache = Cache()
        semantic_cache.init(
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
            pre_embedding_func=get_prompt,
            clusterer=clusterer,
        )

        smart_cache = SmartCache()
        smart_cache.init(
            embedding_func=interceptor,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
            pre_embedding_func=get_prompt
        )

        try:
            for question in test_questions:
                # TODO: implement smart_cache.process_query()
                #smart_cache.process_query()
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
