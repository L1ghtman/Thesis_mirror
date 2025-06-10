import numpy as np
from collections import deque
from typing import Tuple
import json
import datetime
import os

class MagnitudeBasedEstimator:
    """
    Phase 1: Use only embedding magnitude patterns
    """
    def __init__(self, window_size=2000):
        self.magnitude_history = deque(maxlen=window_size)
        self.running_mean = 0.0
        self.running_std = 1.0
        self.total_count = 0
        
    def estimate_density(self, embedding: np.ndarray) -> Tuple[float, dict]:
        """
        Estimate topic density based on magnitude patterns
        Returns: (temperature, debug_info)
        """
        # Calculate magnitude (L2 norm)
        magnitude = np.linalg.norm(embedding) 
        
        # First query - no history
        if self.total_count == 0:
            self.magnitude_history.append(magnitude)
            self.running_mean = magnitude
            self.total_count = 1
            return 1.5, {"magnitude": magnitude, "z_score": 0, "density": 0}
        
        # Calculate z-score (how unusual is this magnitude?)
        z_score = abs(magnitude - self.running_mean) / max(self.running_std, 0.1)
        
        # Density based on how typical the magnitude is
        # Typical magnitudes = high density = low temperature
        if z_score < 0.5:
            density = 0.9  # Very typical
        elif z_score < 1.0:
            density = 0.7  # Somewhat typical
        elif z_score < 2.0:
            density = 0.4  # Somewhat unusual
        else:
            density = 0.1  # Very unusual
        
        # Update statistics
        self._update_stats(magnitude)
        
        # Convert density to temperature
        base_temp = 0.8
        temperature = base_temp * (1 - density * 0.7)
        
        debug_info = {
            "magnitude": round(magnitude, 3),
            "mean_magnitude": round(self.running_mean, 3),
            "z_score": round(z_score, 3),
            "density": round(density, 3),
            "temperature": round(temperature, 3)
        }
        
        return temperature, debug_info
    
    def _update_stats(self, magnitude: float):
        """Update running statistics efficiently"""
        self.magnitude_history.append(magnitude)
        self.total_count += 1
        
        # Update running mean and std
        if len(self.magnitude_history) > 10:
            # Efficient incremental update
            old_mean = self.running_mean
            self.running_mean = np.mean(self.magnitude_history)
            self.running_std = np.std(self.magnitude_history)

# Integration with your system
class MagnitudeCache:
    def __init__(self, debug_log_path="./logs/magnitude_debug.log"):
        self.estimator = MagnitudeBasedEstimator()
        self.base_temperature = 0.8
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
        
        # Open the debug log file
        self.debug_log_path = debug_log_path
        self.debug_log = open(debug_log_path, 'a')
        print(f"Writing magnitude debug info to: {debug_log_path}")
        
    def get_temperature(self, embedding):
        temp, debug = self.estimator.estimate_density(embedding)
        print(f"Debug: {debug}")  # TODO: Remove in production
        timestamp = datetime.datetime.now().isoformat()
        # Convert all values in debug to native Python types
        debug_clean = {k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v for k, v in debug.items()}
        log_entry = {
            "timestamp": timestamp,
            **debug_clean
        }
        self.debug_log.write(json.dumps(log_entry) + "\n")
        self.debug_log.flush()

        # Get magnitude from debug info
        try:
            magnitude = debug.get("magnitude", 0.0)
            print(f"Extracted magnitude: {magnitude}")
        except Exception as e:
            print(f"Error extracting magnitude: {e}")
        
        return temp, magnitude
    
    def __del__(self):
        """Ensure the file is closed when the object is destroyed"""
        if hasattr(self, 'debug_log') and self.debug_log:
            self.debug_log.close()