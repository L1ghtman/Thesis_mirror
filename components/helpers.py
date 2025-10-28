import time

from datetime import datetime
from typing import Tuple
from config_manager import Config

# Llama-3 prompt tokens
BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"
EOT = "<|eot_id|>"

# These wrap around one of the three possible roles: system, user, assistant
SH = "<|start_header_id|>"
EH = "<|end_header_id|>"
# System prompt for testing
# SP = "You are a helpful AI assistant. You are currently being tested. Please only respond with 'This is a test'."
# System prompt for benchmarking
SP = "You are a helpful AI assistant. Provide clear, accurate, and concise responses to user queries. Keep your answers factual and well-structured. Aim for completeness while being efficient with words."
# Take a prompt and format it into Llama-3 prompt token format according to https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
def prompt_format(prompt: str, first_call: bool) -> Tuple[str, bool]:
    """Take a user input and format it into llama3 prompt token format."""
    return (f"{SH}system{EH}\n\n{SP}{EOT}{SH}user{EH}\n\n{prompt}{EOT}{SH}assistant{EH}\n\n", first_call)
   
def convo_aware_prompt_format(prompt: str, first_call: bool) -> Tuple[str, bool]:
    """Take a user input and format it into llama3 prompt token format."""
    if first_call:
        first_call = False
        # TODO: insert '\n\n' into tokens itself
        # return (f"{BOS}{SH}system{EH}\n\n{SP}{EOT}{SH}user{EH}\n\n{prompt}{EOT}{SH}assistant{EH}\n\n", first_call)
        return (f"{SH}system{EH}\n\n{SP}{EOT}{SH}user{EH}\n\n{prompt}{EOT}{SH}assistant{EH}\n\n", first_call)
    else:
        return (f"{prompt}{EOT}{SH}assistant{EH}\n\n", first_call)


def convert_gptcache_report(cache_obj, log_dir="cache_logs"):
    """Convert GPTCache report to format compatible with cache_analyzer"""
    report = cache_obj.report

    full_metrics = {
        "pre_process_time": report.op_pre.total_time,
        "pre_process_count": report.op_pre.count,
        "embedding_time": report.op_embedding.total_time,
        "embedding_count": report.op_embedding.count,
        "clustering_time": report.op_clustering.total_time,
        "clustering_count": report.op_clustering.count,
        "temperature_time": report.op_temperature.total_time,
        "temperature_count": report.op_temperature.count,
        "search_time": report.op_search.total_time,
        "search_count": report.op_search.count,
        "data_time": report.op_data.total_time,
        "data_count": report.op_data.count,
        "eval_time": report.op_evaluation.total_time,
        "eval_count": report.op_evaluation.count,
        "post_process_time": report.op_post.total_time,
        "post_process_count": report.op_post.count,
        "llm_time": report.op_llm.total_time,
        "llm_count": report.op_llm.count,
        "llm_direct_time": report.op_llm_direct.total_time,
        "llm_direct_count": report.op_llm_direct.count,
        "save_time": report.op_save.total_time,
        "save_count": report.op_save.count,
        "average_pre_time": report.op_pre.average(),
        "average_emb_time": report.op_embedding.average(),
        "average_temperature_time": report.op_temperature.average(),
        "average_search_time": report.op_search.average(),
        "average_data_time": report.op_data.average(),
        "average_eval_time": report.op_evaluation.average(),
        "average_post_time": report.op_post.average(),
        "average_llm_time": report.op_llm.average(),
        "average_save_time": report.op_save.average(),
        "cache_hits": report.hint_cache_count,
        }
    
    return full_metrics

def track_request(question, response, start_time, is_cache_hit, similarity_score=None, used_cache=True, temperature=None):
    request_data = {
        "timestamp": datetime.now().isoformat(),
        "query": question,
        "response": response,
        "event_type": "CACHE_HIT" if is_cache_hit else "CACHE_MISS",
        "response_time": time.time() - start_time,
        "similarity_score": similarity_score,
        "used_cache": used_cache,
        "temperature": temperature
    }
    return request_data

def process_request(question, cached_llm, semantic_cache, CacheLogger, use_cache):
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
    report_metrics = convert_gptcache_report(semantic_cache)

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


def get_info_level(config: Config):
    info_level = config.sys['info_level']
    INFO = False
    DEBUG = False
    if info_level == 0:
        INFO = False
        DEBUG = False
    elif info_level == 1:
        INFO = True
        DEBUG = False
    elif info_level == 2:
        INFO = False
        DEBUG = True
    elif info_level == 3:
        INFO = True
        DEBUG = True

    return INFO, DEBUG

def info_print(msg, INFO):
    if INFO:
        print(f'[INFO] {msg}')
 
def debug_print(msg, DEBUG):
    if DEBUG:
        print(f'[DEBUG] {msg}')