import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

class CachePerformanceLogger:
    """
    Logger class for tracking GPTCache performance metrics.
    Creates sequential log files and captures detailed performance data.
    """
    
    def __init__(self, log_dir: str = "cache_logs"):
        """
        Initialize the logger with a directory for storing log files.
        
        Args:
            log_dir: Directory where log files will be stored
        """
        self.log_dir = log_dir
        self.current_run_id = self._get_next_run_id()
        self.log_file = self._create_log_file()
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "positive_hits": 0,
            "negative_hits": 0,
            "llm_direct_calls": 0,
            "cache_response_times": [],
            "llm_response_times": [],
            "requests": []
        }
        
        # Configure Python's logging module
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
        """Determine the next run ID based on existing log files."""
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
        """Create a new log file with a sequential run number."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        log_file = os.path.join(self.log_dir, f"cache_run_{self.current_run_id}.json")
        return log_file

    def log_request(self, 
                   query: str, 
                   used_cache: bool, 
                   cache_hit: Optional[bool] = None,
                   positive_hit: Optional[bool] = None,
                   response: str = "",
                   response_time: float = 0.0,
                   similarity_score: Optional[float] = None,
                   temperature: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log information about a single request.
        
        Args:
            query: The user query
            used_cache: Whether the cache was used for this request
            cache_hit: Whether the cache was hit (if used_cache is True)
            positive_hit: Whether the cache hit was positive (if cache_hit is True)
            response: The response returned
            response_time: Time taken to process the request
            similarity_score: Similarity score (for cache hits)
            temperature: Temperature parameter used
            metadata: Any additional metadata to log
        """
        self.metrics["total_requests"] += 1
        
        # Update counters based on cache usage
        if used_cache:
            if cache_hit:
                self.metrics["cache_hits"] += 1
                self.metrics["cache_response_times"].append(response_time)
                
                if positive_hit:
                    self.metrics["positive_hits"] += 1
                    event_type = "POSITIVE_CACHE_HIT"
                else:
                    self.metrics["negative_hits"] += 1
                    event_type = "NEGATIVE_CACHE_HIT"
            else:
                self.metrics["cache_misses"] += 1
                self.metrics["llm_response_times"].append(response_time)
                event_type = "CACHE_MISS"
        else:
            self.metrics["llm_direct_calls"] += 1
            self.metrics["llm_response_times"].append(response_time)
            event_type = "LLM_DIRECT"
        
        # Log the request details
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "event_type": event_type,
            "used_cache": used_cache,
            "cache_hit": cache_hit,
            "positive_hit": positive_hit,
            "response_time": response_time,
            "similarity_score": similarity_score,
            "temperature": temperature
        }
        
        if metadata:
            request_data["metadata"] = metadata
        
        # Add to metrics and log
        self.metrics["requests"].append(request_data)
        
        # Log to Python logger
        self.logger.info(f"{event_type}: query='{query[:50]}...' time={response_time:.4f}s")
        
        # Save updated metrics to file
        self._save_metrics()
        
    def log_summary(self) -> Dict[str, Any]:
        """Generate and log a summary of the current run."""
        # Calculate summary statistics
        total_time = sum(self.metrics["cache_response_times"] + self.metrics["llm_response_times"])
        avg_cache_time = sum(self.metrics["cache_response_times"]) / len(self.metrics["cache_response_times"]) if self.metrics["cache_response_times"] else 0
        avg_llm_time = sum(self.metrics["llm_response_times"]) / len(self.metrics["llm_response_times"]) if self.metrics["llm_response_times"] else 0
        
        cache_hit_rate = self.metrics["cache_hits"] / self.metrics["total_requests"] if self.metrics["total_requests"] > 0 else 0
        positive_hit_rate = self.metrics["positive_hits"] / self.metrics["cache_hits"] if self.metrics["cache_hits"] > 0 else 0
        
        summary = {
            "run_id": self.current_run_id,
            "end_time": datetime.now().isoformat(),
            "total_requests": self.metrics["total_requests"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "positive_hits": self.metrics["positive_hits"],
            "negative_hits": self.metrics["negative_hits"],
            "llm_direct_calls": self.metrics["llm_direct_calls"],
            "total_time": total_time,
            "avg_cache_time": avg_cache_time,
            "avg_llm_time": avg_llm_time,
            "cache_hit_rate": cache_hit_rate,
            "positive_hit_rate": positive_hit_rate,
            "time_saved": (avg_llm_time - avg_cache_time) * self.metrics["positive_hits"] if avg_llm_time and self.metrics["positive_hits"] else 0
        }
        
        # Update metrics with summary
        self.metrics["summary"] = summary
        self._save_metrics()
        
        # Log summary to Python logger
        self.logger.info(f"Run #{self.current_run_id} Summary:")
        self.logger.info(f"Total Requests: {summary['total_requests']}")
        self.logger.info(f"Cache Hit Rate: {summary['cache_hit_rate']:.2%}")
        self.logger.info(f"Positive Hit Rate: {summary['positive_hit_rate']:.2%}")
        self.logger.info(f"Average Cache Response Time: {summary['avg_cache_time']:.4f}s")
        self.logger.info(f"Average LLM Response Time: {summary['avg_llm_time']:.4f}s")
        self.logger.info(f"Estimated Time Saved: {summary['time_saved']:.4f}s")
        
        return summary
    
    def _save_metrics(self) -> None:
        """Save the current metrics to the log file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def close(self) -> None:
        """Generate summary and close the logger."""
        self.log_summary()
        self.logger.info(f"Closing cache performance log - Run #{self.current_run_id}")