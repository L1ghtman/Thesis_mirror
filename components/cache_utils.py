import sys
import time
import logging
import traceback
import numpy as np
from gptcache.core import Cache
from gptcache.embedding import SBERT, Huggingface
from transformers import AutoTokenizer, AutoModel
from components import cache_analyzer
from config_manager import get_config
from components.helpers import get_info_level, debug_print, info_print
from transformers import AutoTokenizer, AutoModel

config = get_config()
INFO, DEBUG = get_info_level(config)

_embedding_model_cache = {}

def get_sbert_encoder(model_name):
    global _embedding_model_cache
    if model_name not in _embedding_model_cache:
        info_print(f"Loading embedding model: {model_name} (this may take a moment...)", INFO)
        _embedding_model_cache[model_name] = SBERT(model_name)
        info_print(f"Embedding model loaded successfully", INFO)
    return _embedding_model_cache[model_name]

def embedding_func(prompt, extra_param=None):
    encoder = SBERT('all-MiniLM-L6-v2')
    embedding = encoder.to_embeddings(prompt)
    return tuple(embedding)

def custom_embedding_func(prompt, extra_param=None):
    config = get_config()
    model = config.sys.embedding_model
    debug_print(f"using model {model}", DEBUG)
    encoder = get_sbert_encoder(model)
    embedding = encoder.to_embeddings(prompt)
    return tuple(embedding)

def non_normalized_embedding_func(prompt, extra_param=None):
    encoder = SBERT('all-MiniLM-L6-v2')
    embedding = encoder.to_embeddings(prompt, normalize_embeddings=True)
    return tuple(embedding)

def lsh_temperature_func(lsh_cache, embedding):
    """
    Calculate the temperature based on LSH bucket density.
    """
    temperature, debug_info = lsh_cache.estimate_temperature(embedding)
    return temperature, debug_info

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

