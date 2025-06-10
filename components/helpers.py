import time

from datetime import datetime
from typing import Tuple

# Llama-3 prompt tokens
BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"
EOT = "<|eot_id|>"

# These wrap around one of the three possible roles: system, user, assistant
SH = "<|start_header_id|>"
EH = "<|end_header_id|>"
SP = "You are a helpful AI assistant. Limit your response to 5 sentences. Do not include any disclaimers or apologies. Be concise and to the point. If you don't know the answer, say 'I don't know'."

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
