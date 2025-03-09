import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

class SimpleGPTCacheLogger:
    """
    A simple logger for GPTCache performance metrics.
    Rather than intercepting cache methods, this logger provides simple
    methods to log cache events directly at key points in your code.
    """
    
    def __init__(self, log_dir: str = "cache_logs"):
        """
        Initialize the simple logger with a directory for storing log files.
        
        Args:
            log_dir: Directory where log files will be stored
        """
        self.log_dir = log_dir
        self.current_run_id = self._get_next_run_id()
        self.log_file = self._create_log_file()
        
        # Initialize metrics structure
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
        
        # Set up console and file logging
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f"run_{self.current_run_id}.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"CacheRun-{self.current_run_id}")
        self.logger.info(f"Starting new cache performance log - Run #{self.current_run_id}")
        
        # Create an initial empty log file
        self._save_metrics()
    
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
    
    def log_start_request(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log the start of a request and return a context dictionary to track this request.
        
        Args:
            query: The user query
            metadata: Any additional metadata to log
            
        Returns:
            A context dictionary to pass to subsequent logging methods
        """
        self.metrics["total_requests"] += 1
        
        # Create a request tracking context
        context = {
            "timestamp": datetime.now().isoformat(),
            "start_time": time.time(),
            "query": query,
            "metadata": metadata or {},
            "request_id": self.metrics["total_requests"]
        }
        
        self.logger.info(f"Request {context['request_id']} started: '{query[:50]}...'")
        
        return context
    
    def log_cache_check(self, context: Dict[str, Any], cache_hit: bool, 
                       similarity_score: Optional[float] = None,
                       positive_hit: Optional[bool] = None) -> Dict[str, Any]:
        """
        Log a cache check result.
        
        Args:
            context: The request context from log_start_request
            cache_hit: Whether the cache was hit
            similarity_score: Similarity score for cache hit
            positive_hit: Whether the cache hit was positive
            
        Returns:
            Updated context dictionary
        """
        # Update context with cache check results
        context["cache_hit"] = cache_hit
        context["similarity_score"] = similarity_score
        
        if cache_hit:
            self.metrics["cache_hits"] += 1
            context["event_type"] = "CACHE_HIT"
            
            # Determine if it's a positive hit
            if positive_hit is not None:
                context["positive_hit"] = positive_hit
                if positive_hit:
                    self.metrics["positive_hits"] += 1
                    context["event_type"] = "POSITIVE_CACHE_HIT"
                else:
                    self.metrics["negative_hits"] += 1
                    context["event_type"] = "NEGATIVE_CACHE_HIT"
            
            self.logger.info(
                f"Request {context['request_id']}: Cache HIT "
                f"(similarity: {similarity_score:.3f if similarity_score is not None else 'N/A'}, "
                f"positive: {positive_hit})"
            )
        else:
            self.metrics["cache_misses"] += 1
            context["event_type"] = "CACHE_MISS"
            self.logger.info(f"Request {context['request_id']}: Cache MISS")
        
        return context
    
    def log_llm_direct(self, context: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Log a direct LLM call (cache bypass).
        
        Args:
            context: The request context from log_start_request
            temperature: Temperature parameter used
            
        Returns:
            Updated context dictionary
        """
        self.metrics["llm_direct_calls"] += 1
        context["event_type"] = "LLM_DIRECT"
        context["used_cache"] = False
        context["temperature"] = temperature
        
        self.logger.info(f"Request {context['request_id']}: Direct LLM call (temp: {temperature})")
        
        return context
    
    def log_completion(self, context: Dict[str, Any], response: str = "") -> None:
        """
        Log the completion of a request.
        
        Args:
            context: The request context from previous logging methods
            response: The response text
        """
        # Calculate elapsed time
        end_time = time.time()
        response_time = end_time - context.get("start_time", end_time)
        context["response_time"] = response_time

        print(f"response_time: {response_time}")
        
        # Store response time in appropriate list
        if context.get("event_type") == "POSITIVE_CACHE_HIT" or context.get("event_type") == "NEGATIVE_CACHE_HIT":
            self.metrics["cache_response_times"].append(response_time)
        else:
            self.metrics["llm_response_times"].append(response_time)
        
        # Create a complete request record
        request_data = {
            "timestamp": context.get("timestamp", datetime.now().isoformat()),
            "query": context.get("query", ""),
            "event_type": context.get("event_type", "UNKNOWN"),
            "used_cache": context.get("used_cache", True),
            "cache_hit": context.get("cache_hit", False),
            "positive_hit": context.get("positive_hit", None),
            "response_time": response_time,
            "similarity_score": context.get("similarity_score", None),
            "temperature": context.get("temperature", None)
        }
        
        # Add any metadata
        if "metadata" in context and context["metadata"]:
            request_data["metadata"] = context["metadata"]
        
        # Add request to metrics and save
        self.metrics["requests"].append(request_data)
        self._save_metrics()
        
        # Log completion
        self.logger.info(
            f"Request {context.get('request_id', 'unknown')} completed in {response_time:.3f}s: "
            f"({context.get('event_type', 'UNKNOWN')})"
        )
    
    def log_error(self, context: Dict[str, Any], error: Exception) -> None:
        """
        Log an error that occurred during request processing.
        
        Args:
            context: The request context from previous logging methods
            error: The exception that occurred
        """
        # Calculate elapsed time if available
        if "start_time" in context:
            end_time = time.time()
            response_time = end_time - context["start_time"]
            context["response_time"] = response_time
        
        # Create an error request record
        request_data = {
            "timestamp": context.get("timestamp", datetime.now().isoformat()),
            "query": context.get("query", ""),
            "event_type": "ERROR",
            "used_cache": context.get("used_cache", True),
            "cache_hit": context.get("cache_hit", False),
            "positive_hit": context.get("positive_hit", None),
            "response_time": context.get("response_time", 0.0),
            "similarity_score": context.get("similarity_score", None),
            "temperature": context.get("temperature", None),
            "error": str(error)
        }
        
        # Add request to metrics and save
        self.metrics["requests"].append(request_data)
        self._save_metrics()
        
        # Log error
        self.logger.error(
            f"Request {context.get('request_id', 'unknown')} failed: {str(error)}"
        )
    
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
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
    
    def close(self) -> None:
        """Generate summary and close the logger."""
        self.log_summary()
        self.logger.info(f"Closing cache performance log - Run #{self.current_run_id}")