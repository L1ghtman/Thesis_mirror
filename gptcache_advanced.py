import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CACHE_DIR = "./persistent_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

import warnings
warnings.filterwarnings("ignore", message="The method `BaseLLM.__call__` was deprecated")

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

# Configure logging to show full details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def verify_cache(cached_llm, semantic_cache):
    # Run a test query that should be cached
    test_question = "Tell me a joke."
    
    start_time = time.time()
    answer = cached_llm(prompt=test_question, cache_obj=semantic_cache)
    first_time = time.time() - start_time
    
    print(f"First query time: {first_time:.4f}s")

    # Run the same query again - should be faster if cache works
    start_time = time.time()
    answer = cached_llm(prompt=test_question, cache_obj=semantic_cache)
    second_time = time.time() - start_time
    
    print(f"Second query time: {second_time:.4f}s")
    print(f"Speed improvement: {first_time/second_time:.2f}x faster")
    
    return second_time < first_time


def signal_handler(sig, frame):
    print("Benchmark interrupted, generating final report...")
    Cache
    report_path = cache_analyzer.generate_latest_run_report(log_dir="cache_logs")
    print(f"Performance report saved to: {report_path}")
    sys.exit(0)


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

def send_request_to_cache(question, cached_llm, semantic_cache, CacheLogger):
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
    answer = cached_llm(prompt=question, cache_obj=semantic_cache)
    response_time = time.time() - start_time

    is_hit = semantic_cache.report.hint_cache_count > pre_stats["hits"]

    CacheLogger.log_request(
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

def system_cleanup(semantic_cache, data_manager):
    """
    Perform system cleanup tasks.
    """
    # Explicit cleanup in safe order
    semantic_cache.flush()
            
    # Safely cleanup FAISS index
    try:
        if hasattr(data_manager, 'v'):
            logging.debug("Debug: data_manager.v exists")
            logging.debug(f"Debug: data_manager.v attributes: {dir(data_manager.v)}")
            logging.debug(f"Debug: data_manager.v type: {type(data_manager.v)}")

            # Try to access _index and log the result
            try:
                index = getattr(data_manager.v, '_index', None)
                logging.debug(f"Debug: _index value: {index}")
                data_manager.v.flush()
            except Exception as index_error:
                logging.error(f"Error accessing _index: {index_error}")
                logging.error(traceback.format_exc())
    except Exception as e:
        logging.error(f"Warning during FAISS cleanup: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
            
    # Clean up remaining objects
    try:
        del vector_base
        del data_manager
    except Exception as e:
        logging.error(f"Warning during final cleanup: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

def embedding_func(prompt, extra_param=None):
    encoder = SBERT('all-MiniLM-L6-v2')
    embedding = encoder.to_embeddings(prompt)
    #print(f"Embedding dimension: {len(embedding)}")
    return tuple(embedding)

def main():
    try:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout
        )

        evaluation = SbertCrossencoderEvaluation()
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

        
        manager = DatasetManager()
        manager.load_msmarco(split="train", max_samples=None)
        manager.set_active_dataset("msmarco_train")
        questions = manager.get_questions(dataset_name="msmarco_train")

        test_questions = [
            "What is github? Explain briefly.",
            "can you explain what GitHub is? Explain briefly.",
            "can you tell me more about GitHub? Explain briefly.",
            "what is the purpose of GitHub? Explain briefly.",
            "Hello",
            "What is the capital of US?",
            "Tell me a communist joke",
            "Give me a short summary of simulated annealing",
            "What is git cherry pick",
            "Give me a name suggestion for my dog, he likes peanut butter"
        ]

        semantic_cache = Cache()
        semantic_cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
        )

        #semantic_cache.config = Config(similarity_threshold=0.9)

        cached_llm = LangChainLLMs(llm=llm)

        CacheLogger = new_cache_logger.CacheLogger()

        try:
            for question in test_questions:
               send_request_to_cache(question, cached_llm, semantic_cache, CacheLogger) 

            CacheLogger.close()
            report_path = cache_analyzer.generate_latest_run_report(log_dir="cache_logs")
            print(f"Performance report saved to: {report_path}")

        finally:
           system_cleanup(semantic_cache, data_manager)

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
