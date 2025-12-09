import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from config_manager import get_config
from components.helpers import format_time, get_info_level, info_print, debug_print

class LLMLogger:
    def __init__(self, log_dir: str="llm_logs"):
        self.config             = get_config()
        self.INFO, self.DEBUG   = get_info_level(self.config)
        self.log_dir            = log_dir
        self.current_run_id     = self.config.experiment["run_id"]
        self.model              = self.config.sys["model"]
        self.log_file           = self._create_log_file()
        self.metrics            = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm_time": 0,
            "llm_count": 0,
            "average_llm_time": 0,
            "llm_response_times": [],
            "requests": [],
            "summary": {},
        }

#        logging.basicConfig(
#            level=logging.INFO,
#            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#            handlers=[
#                logging.FileHandler(self.log_file),
#                logging.StreamHandler()
#            ]
#        )

        self.logger = logging.getLogger(f"LLMRun-{self.current_run_id}")
        self.logger.info(f"Starting new LLM performance log - Run #{self.current_run_id}")

    def _create_log_file(self) -> str:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        log_file = os.path.join(self.log_dir, f"cache_run_{self.current_run_id}.json")
        return log_file
 
    def log_request(self,
                    query: str,
                    response: str,
                    response_time: float):
        
        reqeust_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "response_time": response_time
        }

        self.metrics["requests"].append(reqeust_data)
        self._save_metrics()

    def log_summary(self) -> Dict[str, Any]:
        response_times = self.metrics["llm_response_times"]
        total_time = sum(response_times)
        avg_llm_time = total_time / len(response_times) if response_times else 0

        summary = {
            "run_id": self.run_id,
            "model": self.model,
            "total_requests": self.total_requests,
            "avg_llm_time": avg_llm_time,
            "total_time": self.total_time
        }

        self.metrics["summary"] = summary
        self._save_metrics()

        self.logger.info(f"Run #{self.current_run_id} Summary: {summary}")
        self.logger.info(f"Total requests: {len(response_times)}")

    def _to_serializable(self, obj):
        """Recursively convert numpy and other non-serializable types to native Python types."""
        if isinstance(obj, dict):
            return {self._to_serializable(k): self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._to_serializable(i) for i in obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif hasattr(obj, 'item') and callable(obj.item):
            # Handles numpy scalar types
            try:
                return obj.item()
            except Exception:
                return str(obj)
        else:
            return obj

    def _save_metrics(self):
        with open(self.log_file, "w") as file:
            json.dump(self._to_serializable(self.metrics), file, indent=2)

    def close(self) -> None:
        self.log_summary()
        self.logger.info(f"Closing LLM log - Run #{self.current_run_id}")