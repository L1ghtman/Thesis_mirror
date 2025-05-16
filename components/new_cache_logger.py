import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

class CacheLogger:

    def __init__(self, log_dir: str="cache_logs"):
        self.log_dir = log_dir
        self.current_run_id = self._get_next_run_id()
        self.log_file = self._create_log_file()
        self.metrics = {
            "pre_process_time": 0, 
            "pre_process_count": 0,
            "embedding_time": 0, 
            "embedding_count": 0,
            "clustering_count": 0,
            "clustering_time": 0,
            "search_time": 0,
            "search_count": 0,
            "data_time": 0, 
            "data_count": 0, 
            "eval_time": 0, 
            "eval_count": 0, 
            "post_process_time": 0,
            "post_process_count": 0, 
            "llm_time": 0, 
            "llm_count": 0, 
            "save_time": 0, 
            "save_count": 0,
            "llm_direct_calls": 0,
            "average_pre_time": 0, 
            "average_emb_time": 0, 
            "average_search_time": 0, 
            "average_data_time": 0, 
            "average_eval_time": 0, 
            "average_post_time": 0, 
            "average_llm_time": 0, 
            "average_save_time": 0, 
            "cache_hits": 0,
            "cache_response_times": [],
            "llm_response_times": [],
            "clustering_times": [],
            "requests": [],
            "summary": {}
        }

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"CacheRun-{self.current_run_id}")
        self.logger.info(f"Starting new cache performance log - Run #{self.current_run_id}")

    def _get_next_run_id(self) -> int:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            return 1
        
        existing_logs = [f for f in os.listdir(self.log_dir)
                            if f.startswith("cache_run_") and f.endswith(".json")]
        
        if not existing_logs:
            return 1
        
        run_ids = [int(f.split("_")[2].split(".")[0]) for f in existing_logs]
        return max(run_ids) + 1 if run_ids else 1
    
    def _create_log_file(self) -> str:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        log_file = os.path.join(self.log_dir, f"cache_run_{self.current_run_id}.json")
        return log_file
    
    def log_request(self, 
                    query: str,
                    response: str,
                    is_cache_hit: bool,
                    response_time: float,
                    used_cache: bool,
                    similarity_score: Optional[float] = None,
                    temperature: Optional[float] = None,
                    cluster_id: Optional[int] = None,
                    report_metrics: Dict[str, Any] = {}):
        
        self.metrics["clustering_times"].append(report_metrics.get("clustering_time", 0))
        
        print(f"cluster times: {self.metrics['clustering_times']}")

        if used_cache:
            if is_cache_hit:
                self.metrics["cache_response_times"].append(response_time)
                event_type = "CACHE_HIT"
                # TODO: add positive/negative hit tracking
            else:
                self.metrics["llm_response_times"].append(response_time)
                event_type = "CACHE_MISS"
        else:
            self.metrics["llm_direct_calls"] += 1
            self.metrics["llm_response_times"].append(response_time)
            event_type = "LLM_DIRECT_CALL"

        if cluster_id is None and report_metrics.get("semantic_cache") is not None:
            if hasattr(report_metrics["semantic_cache"], "last_cluster_id"):
                cluster_id = report_metrics["semantic_cache"].last_cluster_id

        request_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "event_type": event_type,
            "response_time": response_time,
            "similarity_score": similarity_score,
            "used_cache": used_cache,
            "temperature": temperature
        }

        if cluster_id is not None:
            request_data["cluster_id"] = cluster_id
            print(f"Logging request with cluster_id: {cluster_id}")        

        self.metrics["requests"].append(request_data)

        for key, value in report_metrics.items():
            self.metrics[key] = value

        self.logger.info(f"{event_type}: query='{query[:50]}...' time={response_time:.4f}s cluster_id={cluster_id}")

        self._save_metrics()

    def log_summary(self) -> Dict[str, Any]:
        cache_hit_rate = self.metrics["cache_hits"] / self.metrics["pre_process_count"]

        total_time = sum(self.metrics["cache_response_times"]) + sum(self.metrics["llm_response_times"])
        avg_cache_time = sum(self.metrics["cache_response_times"]) / len(self.metrics["cache_response_times"]) if self.metrics["cache_response_times"] else 0
        avg_llm_time = sum(self.metrics["llm_response_times"]) / len(self.metrics["llm_response_times"]) if self.metrics["llm_response_times"] else 0
        avg_cluster_time = sum(self.metrics["clustering_times"]) / len(self.metrics["clustering_times"]) if self.metrics["clustering_times"] else 0

        print(f"avg cluster times: {avg_cluster_time}")

        avg_embedding_time = self.metrics["embedding_time"] / self.metrics["embedding_count"] if self.metrics["embedding_count"] else 0
        avg_search_time = self.metrics["search_time"] / self.metrics["search_count"] if self.metrics["search_count"] else 0

        summary = {
            "run_id": self.current_run_id,
            "total_requests": self.metrics["pre_process_count"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["pre_process_count"] - self.metrics["cache_hits"],
            "cache_hit_rate": cache_hit_rate,
            "positive_hits": self.metrics["cache_hits"],
            "negative_hits": 0,
            "positive_hit_rate": cache_hit_rate,
            "llm_direct_calls": self.metrics["llm_direct_calls"],
            "avg_cache_time": avg_cache_time,
            "avg_llm_time": avg_llm_time,
            "avg_cluster_time": avg_cluster_time,
            "avg_embedding_time": avg_embedding_time,
            "avg_search_time": avg_search_time,
            "total_time": total_time,
            "time_saved": (avg_llm_time - avg_cache_time) * self.metrics["cache_hits"] if avg_llm_time and self.metrics["cache_hits"] else 0
        }

        self.metrics["summary"] = summary
        self._save_metrics()

        self.logger.info(f"Run #{self.current_run_id} Summary: {summary}")
        self.logger.info(f"Total requests: {self.metrics['pre_process_count']}")
        self.logger.info(f"Cache hit rate: {summary['cache_hit_rate']:.2%}")

        return summary

    def _save_metrics(self):
        with open(self.log_file, "w") as file:
            json.dump(self.metrics, file, indent=2)

    def close(self) -> None:
        self.log_summary()
        self.logger.info(f"Closing cache log - Run #{self.current_run_id}")