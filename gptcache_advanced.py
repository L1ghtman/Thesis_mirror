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

import faiss

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

def verify_cache(cached_llm, llm_cache):
    # Run a test query that should be cached
    test_question = "Tell me a joke."
    
    start_time = time.time()
    answer = cached_llm(prompt=test_question, cache_obj=llm_cache)
    first_time = time.time() - start_time
    
    print(f"First query time: {first_time:.4f}s")

    #intersting_metrics = ["pre_process_count", "embedding_count", "search_count", "data_count", "eval_count", "post_process_count", "llm_count", "save_count", "cache_hits"]
    #metrics = helpers.convert_gptcache_report(llm_cache)
    #for key, value in metrics.items():
    #    if key in intersting_metrics:
    #        print(f"{key}: {value}")


    # Run the same query again - should be faster if cache works
    start_time = time.time()
    answer = cached_llm(prompt=test_question, cache_obj=llm_cache)
    second_time = time.time() - start_time
    
    print(f"Second query time: {second_time:.4f}s")
    
    #intersting_metrics = ["pre_process_count", "embedding_count", "search_count", "data_count", "eval_count", "post_process_count", "llm_count", "save_count", "cache_hits"]
    #metrics = helpers.convert_gptcache_report(llm_cache)
    #for key, value in metrics.items():
    #    if key in intersting_metrics:
    #        print(f"{key}: {value}")
            
    print(f"Speed improvement: {first_time/second_time:.2f}x faster")
    
    return second_time < first_time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Embedding
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Evaluation
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        evaluation = SbertCrossencoderEvaluation()
        #evaluation = custom_sim_eval.CustomSimilarityEvaluation()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Data Manager
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# LLM
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        llm = custom_llm.localLlama()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Questions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        questions = [
            "What is github? Explain briefly.",
            "can you explain what GitHub is? Explain briefly.",
            "can you tell me more about GitHub? Explain briefly.",
            "what is the purpose of GitHub? Explain briefly.",
            "Hello",
            #"What is the capital of US?",
            #"Tell me a pirate joke",
            #"Give me a short summary of simulated annealing",
            #"What is git cherry pick",
            #"Give me a name suggestion for my dog, he likes peanut butter"
        ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Cache initialization
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        llm_cache = Cache()
        llm_cache.init(
            pre_embedding_func=get_prompt,
            embedding_func=embedding_func,
            data_manager=data_manager,
            similarity_evaluation=evaluation,
        )

        #llm_cache.config = Config(similarity_threshold=0.9)

        cached_llm = LangChainLLMs(llm=llm)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Run LLM with cache
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        requests_data = []

        CacheLogger = new_cache_logger.CacheLogger()

        try:
            #print("Cache verification result:", verify_cache(cached_llm, llm_cache))
            for question in questions:
                
                pre_stats = {
                    "hits": llm_cache.report.hint_cache_count,
                    "llm_calls": llm_cache.report.op_llm.count,
                }
                
                start_time = time.time()
                answer = cached_llm(prompt=question, cache_obj=llm_cache)
                response_time = time.time() - start_time

                is_hit = llm_cache.report.hint_cache_count > pre_stats["hits"]

                #requests_data.append(helpers.track_request(question, answer, start_time, is_hit))

                CacheLogger.log_request(
                    query=question,
                    response=answer,
                    response_time=response_time,
                    is_cache_hit=is_hit,
                    similarity_score=None,
                    used_cache=True,
                    temperature=None,
                    report_metrics=helpers.convert_gptcache_report(llm_cache)
                )

                print(f"Question: {question}")
                print("Time consuming: {:.2f}s".format(time.time() - start_time))
                print(f"Answer: {answer}\n")
                # print this line in blue color
                print("\033[94m" + "-----------------------------------------------------------" + "\033[0m\n")

            CacheLogger.close()

            report_path = cache_analyzer.generate_latest_run_report(log_dir="cache_logs")
            print(f"Performance report saved to: {report_path}")

            #intersting_metrics = ["pre_process_count", "embedding_count", "search_count", "data_count", "eval_count", "post_process_count", "llm_count", "save_count", "cache_hits"]
            #metrics = helpers.convert_gptcache_report(llm_cache)
            #for key, value in metrics.items():
            #    if key in intersting_metrics:
            #        print(f"{key}: {value}")
            
            #metrics["requests"] = requests_data

            #print("Requests data:")
            #for request in requests_data:
            #    for key, value in request.items():
            #        print(f"{key}: {value}")
            #    # print green line
            #    print("\033[92m" + "-----------------------------------------------------------" + "\033[0m\n")
        finally:
            # Explicit cleanup in safe order
            llm_cache.flush()
            
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
                    
                    #if hasattr(data_manager.v, '_index') and data_manager.v._index is not None:
                    #    logging.debug("Debug: _index exists and is not None")
                    #    if hasattr(data_manager.v, '_index_file_path'):
                    #        logging.debug(f"Debug: _index_file_path exists: {data_manager.v._index_file_path}")
                    #        faiss.write_index(data_manager.v._index, data_manager.v._index_file_path)
                    #    data_manager.v._index.reset()
                        #del data_manager.v._index
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

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
