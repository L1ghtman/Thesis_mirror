import json
import time
import math
import numpy as np
from typing import Tuple, List, Dict, Any, Set
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from config_manager import get_config
from components.helpers import get_info_level, info_print, debug_print

def get_layered_hyperplanes(lsh_layers, num_hyperplanes, embedding_dim):
    layered_hyperplanes = []
    for _ in range(lsh_layers):
        hyperplanes = np.random.randn(num_hyperplanes, embedding_dim)
        hyperplanes /= np.linalg.norm(hyperplanes, axis=1, keepdims=True)
        layered_hyperplanes.append(hyperplanes)
    return layered_hyperplanes

class LSHEstimator:
    def __init__(self, embedding_dim, num_hyperplanes, lsh_layers, window_size, config):
        # Original LSH components
        self.config = config
        self.num_hyperplanes = num_hyperplanes
        self.hyperplanes = np.random.randn(num_hyperplanes, embedding_dim)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)
        self.layered_hyperplanes = get_layered_hyperplanes(lsh_layers, num_hyperplanes, embedding_dim)
        self.bucket_counts = defaultdict(int)
        self.bucket_history = deque(maxlen=window_size)
        self.total_count = 0
        self.bucket_density_factor = self.config.experiment.bucket_density_factor
        self.sensitivity = self.config.experiment.sensitivity
        self.decay_rate = self.config.experiment.decay_rate
        
        # Enhanced tracking components
        self.tracking_data = {
            'bucket_sequences': [],  # Sequence of bucket IDs
            'bucket_densities': defaultdict(list),  # Density history per bucket
            'bucket_temperatures': defaultdict(list),  # Temperature history per bucket
            'bucket_first_seen': {},  # Timestamp when bucket first appeared
            'bucket_last_seen': {},  # Timestamp when bucket last accessed
            'neighbor_contributions': [],  # How much neighbors contributed to density
            'hamming_distances': [],  # Distance between consecutive buckets
            'query_embeddings': [],  # Store embeddings for analysis
            'timestamps': [],  # Timestamp for each query
            'bucket_transitions': defaultdict(lambda: defaultdict(int)),  # Transition matrix
            'bucket_hit_quality': defaultdict(list),  # Track cache hit quality per bucket
        }
        
        # Performance tracking
        self.performance_metrics = {
            'density_calculation_times': [],
            'bucket_assignment_times': [],
            'total_buckets_created': 0,
            'bucket_reuse_rate': 0,
            'average_bucket_lifetime': 0,
        }
        
        # Item-to-bucket tracking for eviction synchronization
        self.item_to_bucket: Dict[int, str] = {}  # item_id → bucket_id
        self.bucket_to_items: Dict[str, Set[int]] = defaultdict(set)  # bucket_id → {item_ids}
        self.active_item_count = 0  # Track items that haven't been evicted
        
        # Cache last computed bucket to avoid redundant computation
        self.last_bucket: str = None
        self.last_bucket_embedding_hash: int = None  # To verify it's the same embedding

        INFO, DEBUG = get_info_level(self.config)

        info_print('--- LSHEstimator configuration ---', INFO)
        info_print(f"bucket_density_factor:     {self.bucket_density_factor}", INFO)
        info_print(f"sensitivity:               {self.sensitivity}", INFO)
        info_print(f"decay_rate:                {self.decay_rate}", INFO)
        info_print('----------------------------------', INFO)

    def get_lsh_bucket(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Hash embedding to bucket using hyperplanes, return bucket and computation time"""
        start_time = time.time()
        projections = np.dot(self.hyperplanes, embedding)
        binary = ''.join('1' if p > 0 else '0' for p in projections)
        computation_time = time.time() - start_time
        self.performance_metrics['bucket_assignment_times'].append(computation_time)
        return binary, computation_time       

    def get_layered_lsh_bucket(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Hash embedding to several buckets using layered hyperplanes, return buckets and computation time"""
        start_time = time.time()
        binaries = []
        for layer in self.layered_hyperplanes:
            projections = np.dot(layer, embedding)
            binary = ''.join('1' if p > 0 else '0' for p in projections)
            binaries.append(binary)
        computation_time = time.time() - start_time
        self.performance_metrics['bucket_assignment_times'].append(computation_time)
        return binary, computation_time
    
    def estimate_density(self, embedding: np.ndarray, lsh_layers) -> Tuple[float, dict]:
        """
        Estimate topic density using LSH with comprehensive tracking
        Returns: (temperature, debug_info)
        """

        timestamp = datetime.now()
        
        # Get bucket(s)
        if lsh_layers > 1:
            layered_buckets = self.get_layered_lsh_bucket(embedding) #TODO: integrate layered LSH architecture
        else:
            bucket, bucket_time = self.get_lsh_bucket(embedding)
        
        # Cache the bucket for potential reuse in register_item()
        self.last_bucket = bucket
        self.last_bucket_embedding_hash = hash(embedding.tobytes()) if hasattr(embedding, 'tobytes') else hash(tuple(embedding))
        
        # Track bucket sequence and transitions
        if self.bucket_history:
            last_bucket = self.bucket_history[-1]
            hamming_dist = sum(c1 != c2 for c1, c2 in zip(bucket, last_bucket))
            self.tracking_data['hamming_distances'].append(hamming_dist)
            self.tracking_data['bucket_transitions'][last_bucket][bucket] += 1
        
        # Calculate density with tracking
        start_density_time = time.time()
        bucket_density, neighbor_contribution = self._calculate_bucket_density_with_tracking(bucket)
        density_calc_time = time.time() - start_density_time
        self.performance_metrics['density_calculation_times'].append(density_calc_time)
        
        density = bucket_density * self.bucket_density_factor
        
        # Update tracking before updating stats
        self._track_bucket_access(bucket, density, timestamp, embedding)
        self.tracking_data['neighbor_contributions'].append(neighbor_contribution)
        
        # Update stats
        self._update_stats(bucket)

        curve = self.config.experiment.curve

        if curve == "exponential":
            # Exponential with sensitivity
            sensitivity = self.config.experiment.sensitivity
            temperature = 2.0 * math.exp(-sensitivity * density)
        else:
            # Rational with decay rate
            decay_rate = self.config.experiment.decay_rate
            temperature = 2.0 / (1 + decay_rate * density)
        
        # Track temperature for this bucket
        self.tracking_data['bucket_temperatures'][bucket].append(temperature)
        self.tracking_data['bucket_densities'][bucket].append(density)
        
        #cache_factor = 3.141

        # Enhanced debug info
        debug_info = {
            "lsh_bucket": bucket,
            "bucket_count": self.bucket_counts[bucket],
            "bucket_density": round(bucket_density, 3),
            "neighbor_contribution": round(neighbor_contribution, 3),
            "density": round(density, 3),
            "temperature": round(temperature, 3),
            #"cache_factor": round(cache_factor, 3),
            "bucket_age": self._get_bucket_age(bucket),
            "total_unique_buckets": len(self.bucket_counts),
            "bucket_reuse_rate": self._calculate_bucket_reuse_rate(),
            "hamming_distance_from_last": self.tracking_data['hamming_distances'][-1] if self.tracking_data['hamming_distances'] else 0,
            "computation_time_ms": round((bucket_time + density_calc_time) * 1000, 2)
        }
        
        return temperature, debug_info
    
    def _calculate_bucket_density_with_tracking(self, bucket: str) -> Tuple[float, float]:
        """Calculate density for a bucket and its neighbors, returning both total density and neighbor contribution"""
        if self.total_count == 0:
            return 0.0, 0.0
            
        main_count = self.bucket_counts[bucket]
        
        neighbor_count = 0
        for i in range(len(bucket)):
            neighbor = list(bucket)
            neighbor[i] = '1' if bucket[i] == '0' else '0'
            neighbor_key = ''.join(neighbor)
            neighbor_count += self.bucket_counts.get(neighbor_key, 0)
        
        relevant_count = main_count + neighbor_count * 0.5  # Neighbors weighted less
        
        density = min(relevant_count / (self.total_count * 0.1), 1.0) # clamping density to stay between 1.0 and 0.0. The '*0.1' is used to configure the % of total items in a bucket and its neighbors to trigger maximum density output
        neighbor_contribution = (neighbor_count * 0.5) / max(relevant_count, 1)
        
        return density, neighbor_contribution
    
    def _track_bucket_access(self, bucket: str, density: float, timestamp: datetime, embedding: np.ndarray):
        """Track comprehensive bucket access information"""
        # Track first and last seen
        if bucket not in self.tracking_data['bucket_first_seen']:
            self.tracking_data['bucket_first_seen'][bucket] = timestamp
            self.performance_metrics['total_buckets_created'] += 1
        self.tracking_data['bucket_last_seen'][bucket] = timestamp
        
        # Store tracking data
        self.tracking_data['bucket_sequences'].append(bucket)
        self.tracking_data['timestamps'].append(timestamp.isoformat())
        
        # Handle the case where embedding might be a tuple or already a list
        if isinstance(embedding, tuple):
            self.tracking_data['query_embeddings'].append(list(embedding))
        elif hasattr(embedding, 'tolist'):
            self.tracking_data['query_embeddings'].append(embedding.tolist())
        else:
            # Convert to list if it's another iterable type
            self.tracking_data['query_embeddings'].append(list(embedding) if hasattr(embedding, '__iter__') else [embedding])
        return 0
    
    def _get_bucket_age(self, bucket: str) -> float:
        """Get age of bucket in seconds"""
        if bucket not in self.tracking_data['bucket_first_seen']:
            return 0.0
        age = (datetime.now() - self.tracking_data['bucket_first_seen'][bucket]).total_seconds()
        return round(age, 2)
    
    def _calculate_bucket_reuse_rate(self) -> float:
        """Calculate what percentage of queries reuse existing buckets"""
        if self.total_count == 0:
            return 0.0
        unique_buckets = len(self.bucket_counts)
        reuse_rate = 1.0 - (unique_buckets / self.total_count)
        return round(max(0.0, reuse_rate), 3)
    
    def _update_stats(self, bucket: str):
        """Update all statistics"""
        self.bucket_history.append(bucket)
        self.bucket_counts[bucket] += 1
        self.total_count += 1
        
        # Update performance metrics
        self.performance_metrics['bucket_reuse_rate'] = self._calculate_bucket_reuse_rate()
    
    def register_item(self, item_id: int, embedding: np.ndarray = None, bucket: str = None) -> str:
        """
        Register a cache item with its LSH bucket.
        Called when a new item is added to the cache.
        
        Args:
            item_id: The unique ID of the cached item (from scalar storage)
            embedding: The embedding vector for this item (optional if bucket is provided)
            bucket: Pre-computed bucket ID (optional, avoids recomputation)
            
        Returns:
            The bucket ID this item was assigned to
        """
        # Use provided bucket, or cached bucket, or compute it
        if bucket is None:
            if embedding is not None:
                # Check if we can reuse the cached bucket
                embedding_hash = hash(embedding.tobytes()) if hasattr(embedding, 'tobytes') else hash(tuple(embedding))
                if self.last_bucket is not None and self.last_bucket_embedding_hash == embedding_hash:
                    bucket = self.last_bucket
                else:
                    bucket, _ = self.get_lsh_bucket(embedding)
            elif self.last_bucket is not None:
                # Fall back to last cached bucket
                bucket = self.last_bucket
            else:
                raise ValueError("Either embedding or bucket must be provided")

        print("Item registered!")
        
        # Store bidirectional mapping
        self.item_to_bucket[item_id] = bucket
        self.bucket_to_items[bucket].add(item_id)
        self.active_item_count += 1
        
        return bucket
    
    def evict_item(self, item_id: int) -> bool:
        """
        Handle eviction of a cache item - decrement bucket count.
        Called when an item is evicted from the cache.
        
        Args:
            item_id: The unique ID of the evicted item
            
        Returns:
            True if item was found and evicted, False otherwise
        """
        bucket = self.item_to_bucket.get(item_id)
        
        print("evicting item")

        if bucket is None:
            print("fail")
            # Item wasn't registered (might be from before LSH tracking)
            return False
        
        # Decrement bucket count
        if self.bucket_counts[bucket] > 0:
            self.bucket_counts[bucket] -= 1
        
        # Update total count
        if self.total_count > 0:
            self.total_count -= 1
        
        # Remove from tracking structures
        self.bucket_to_items[bucket].discard(item_id)
        del self.item_to_bucket[item_id]
        self.active_item_count -= 1
        
        # Clean up empty bucket sets
        if len(self.bucket_to_items[bucket]) == 0:
            del self.bucket_to_items[bucket]
        
        print("success")
        return True
    
    def evict_items(self, item_ids: List[int]) -> int:
        """
        Batch evict multiple items.
        
        Args:
            item_ids: List of item IDs to evict
            
        Returns:
            Number of items successfully evicted
        """
        evicted_count = 0
        for item_id in item_ids:
            if self.evict_item(item_id):
                evicted_count += 1
        return evicted_count
    
    def get_active_bucket_stats(self) -> Dict[str, Any]:
        """Get statistics about active (non-evicted) items per bucket."""
        return {
            'total_active_items': self.active_item_count,
            'total_tracked_items': len(self.item_to_bucket),
            'buckets_with_active_items': len(self.bucket_to_items),
            'items_per_bucket': {b: len(items) for b, items in self.bucket_to_items.items()},
        }
    
    def save_tracking_data(self, filepath: str):
        """Save tracking data to JSON file for analysis"""
        export_data = {
            'configuration': {
                'num_hyperplanes': self.num_hyperplanes,
                'embedding_dim': len(self.hyperplanes[0]),
                'total_queries': self.total_count,
                'unique_buckets': len(self.bucket_counts),
            },
            'bucket_statistics': {
                'bucket_counts': dict(self.bucket_counts),
                'bucket_first_seen': {k: v.isoformat() for k, v in self.tracking_data['bucket_first_seen'].items()},
                'bucket_last_seen': {k: v.isoformat() for k, v in self.tracking_data['bucket_last_seen'].items()},
            },
            'sequences': {
                'bucket_sequences': self.tracking_data['bucket_sequences'],
                'timestamps': self.tracking_data['timestamps'],
                'hamming_distances': self.tracking_data['hamming_distances'],
                'neighbor_contributions': self.tracking_data['neighbor_contributions'],
            },
            'performance': self.performance_metrics,
            'transitions': dict(self.tracking_data['bucket_transitions']),
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_bucket_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about bucket usage"""
        if not self.bucket_counts:
            return {}
        
        bucket_access_counts = list(self.bucket_counts.values())
        bucket_lifetimes = []
        
        for bucket in self.bucket_counts:
            if bucket in self.tracking_data['bucket_first_seen'] and bucket in self.tracking_data['bucket_last_seen']:
                lifetime = (self.tracking_data['bucket_last_seen'][bucket] - 
                           self.tracking_data['bucket_first_seen'][bucket]).total_seconds()
                bucket_lifetimes.append(lifetime)
        
        analytics = {
            'total_buckets': len(self.bucket_counts),
            'total_queries': self.total_count,
            'bucket_reuse_rate': self._calculate_bucket_reuse_rate(),
            'most_accessed_buckets': sorted(self.bucket_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'access_distribution': {
                'mean': np.mean(bucket_access_counts),
                'median': np.median(bucket_access_counts),
                'std': np.std(bucket_access_counts),
                'max': max(bucket_access_counts),
                'min': min(bucket_access_counts),
            },
            'lifetime_stats': {
                'mean_lifetime': np.mean(bucket_lifetimes) if bucket_lifetimes else 0,
                'max_lifetime': max(bucket_lifetimes) if bucket_lifetimes else 0,
            },
            'hamming_distance_stats': {
                'mean': np.mean(self.tracking_data['hamming_distances']) if self.tracking_data['hamming_distances'] else 0,
                'std': np.std(self.tracking_data['hamming_distances']) if self.tracking_data['hamming_distances'] else 0,
            }
        }
        
        return analytics


class LSHCache:
    def __init__(self):

        self.config = get_config()
        self.dimension = self.config.vector_store.dimension
        self.use_LSH = self.config.experiment.use_LSH
        self.num_hyperplanes = self.config.experiment.num_hyperplanes
        self.lsh_layers = self.config.experiment.layers
        self.window_size = self.config.experiment.window_size

        self.estimator = LSHEstimator(self.dimension,
                                      self.num_hyperplanes, 
                                      self.lsh_layers,
                                      self.window_size,
                                      self.config
                                    )
        self.last_debug_info = None  # Store last debug info

        INFO, DEBUG = get_info_level(self.config)

        info_print('--- LSHCache configuration ---', INFO)
        info_print(f"use_LSH:                   {self.use_LSH}", INFO)
        info_print(f"embedding dimension:       {self.dimension}", INFO)
        info_print(f"num_hyperplanes:           {self.num_hyperplanes}", INFO)
        info_print(f"lsh layers:                {self.lsh_layers}", INFO)
        info_print(f"window size:               {self.window_size}", INFO)
        info_print('----------------------------------', INFO)
    
    def estimate_temperature(self, embedding: np.ndarray) -> Tuple[float, dict]:
        temperature, debug_info = self.estimator.estimate_density(embedding, self.lsh_layers)
        self.last_debug_info = debug_info  # Store for access
        return temperature, debug_info
    
    def get_temperature(self, embedding: np.ndarray) -> float:
        temperature, debug_info = self.estimate_temperature(embedding)
        return temperature
    
    def get_last_debug_info(self) -> dict:
        """Get the last debug info for logging"""
        return self.last_debug_info
    
    def register_item(self, item_id: int, embedding: np.ndarray = None, bucket: str = None) -> str:
        """
        Register a cached item with its LSH bucket.
        
        Args:
            item_id: The unique ID of the cached item
            embedding: The embedding vector (optional if bucket provided or cached)
            bucket: Pre-computed bucket ID from debug_info['lsh_bucket'] (optional)
        """
        return self.estimator.register_item(item_id, embedding, bucket)
    
    def evict_item(self, item_id: int) -> bool:
        """Handle eviction of a single item."""
        return self.estimator.evict_item(item_id)
    
    def evict_items(self, item_ids: List[int]) -> int:
        """Handle batch eviction of items."""
        return self.estimator.evict_items(item_ids)
    
    def get_active_stats(self) -> Dict[str, Any]:
        """Get active item statistics."""
        return self.estimator.get_active_bucket_stats()