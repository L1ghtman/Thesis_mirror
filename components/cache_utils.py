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
import numpy as np

def embedding_func(prompt, extra_param=None):
    encoder = SBERT('all-MiniLM-L6-v2')
    embedding = encoder.to_embeddings(prompt)
    return tuple(embedding)

def temperature_func(clusterer, embedding_data, cache_size, temperature):
    """
    Calculate the temperature based on the clustering and cache size.
    """
    clusterer.add_to_buffer(embedding_data)
    clusterer.process_buffer()
    cache_factor = 1.0/max(1, np.log2(cache_size+1))
    cluster_adjustment = clusterer.get_temperature_adjustment(embedding_data)
    original_temperature = temperature
    adjusted_temperature = original_temperature * (0.4*cache_factor + 0.6*cluster_adjustment)
    temperature = max(0.0, min(2.0, adjusted_temperature))
    print(f"Temp adjustment: {original_temperature:.2f} → {temperature:.2f} " +
              f"(cluster: {cluster_adjustment:.2f}, cache: {cache_factor:.2f})")
    print(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .")
    return temperature

def system_cleanup(semantic_cache, vector_base, data_manager):
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

