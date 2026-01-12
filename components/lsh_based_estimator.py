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

def get_hamming_distance(a, b):
    return sum(c1 != c2 for c1, c2 in zip(a, b))

class LSHEstimator:
    def __init__(self, embedding_dim, num_hyperplanes, lsh_layers, window_size, config):
        # Original LSH components
        self.config = config
        self.num_hyperplanes = num_hyperplanes
        self.lsh_layers = lsh_layers
        self.window_size = window_size
        self.hyperplanes = np.random.randn(num_hyperplanes, embedding_dim)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)
        self.layered_hyperplanes = get_layered_hyperplanes(lsh_layers, num_hyperplanes, embedding_dim)
        
        # Per-layer data structures
        self.bucket_counts: List[Dict[str, int]] = [defaultdict(int) for _ in range(lsh_layers)]
        self.bucket_history: List[deque] = [deque(maxlen=window_size) for _ in range(lsh_layers)]
        
        self.total_count = 0  # Global query count
        self.bucket_density_factor = self.config.experiment.bucket_density_factor
        self.sensitivity = self.config.experiment.sensitivity
        self.decay_rate = self.config.experiment.decay_rate
        self.knn_density = self.config.experiment.knn_density
        self.k = self.config.experiment.k
        
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
        
        # Item-to-bucket tracking for eviction synchronization (multi-layer)
        self.item_to_bucket: Dict[int, List[str]] = {}  # item_id → [bucket_id per layer]
        self.bucket_to_items: List[Dict[str, Set[int]]] = [defaultdict(set) for _ in range(lsh_layers)]  # layer → bucket_id → {item_ids}
        self.active_item_count = 0  # Track items that haven't been evicted
        
        # Cache last computed buckets to avoid redundant computation (multi-layer)
        self.last_buckets: List[str] = None  # One bucket per layer
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

    def get_layered_lsh_bucket(self, embedding: np.ndarray) -> Tuple[List[str], float]:
        """Hash embedding to several buckets using layered hyperplanes, return list of buckets and computation time"""
        start_time = time.time()
        binaries = []
        for layer in self.layered_hyperplanes:
            projections = np.dot(layer, embedding)
            binary = ''.join('1' if p > 0 else '0' for p in projections)
            binaries.append(binary)
        computation_time = time.time() - start_time
        self.performance_metrics['bucket_assignment_times'].append(computation_time)
        return binaries, computation_time
    
    def estimate_density(self, embedding: np.ndarray, lsh_layers) -> Tuple[float, dict]:
        """
        Estimate topic density using multi-layer LSH with comprehensive tracking.
        Computes density for each layer and averages them.
        
        Returns: (temperature, debug_info)
        """
        timestamp = datetime.now()
        
        # Get buckets for all layers
        buckets, bucket_time = self.get_layered_lsh_bucket(embedding)
        
        # Cache the buckets for potential reuse in register_item()
        self.last_buckets = buckets
        self.last_bucket_embedding_hash = hash(embedding.tobytes()) if hasattr(embedding, 'tobytes') else hash(tuple(embedding))
        
        # Track bucket sequence and transitions (using layer 0 as reference)
        # TODO: evaluate if this is needed at all
        if self.bucket_history[0]:
            last_bucket = self.bucket_history[0][-1]
            hamming_dist = get_hamming_distance(buckets[0], last_bucket)
            self.tracking_data['hamming_distances'].append(hamming_dist)
            self.tracking_data['bucket_transitions'][last_bucket][buckets[0]] += 1
        
        # Calculate density averaged across all layers
        start_density_time = time.time()
        bucket_density, neighbor_contribution, per_layer_densities = self._calculate_bucket_density_layered(buckets)
        density_calc_time = time.time() - start_density_time
        self.performance_metrics['density_calculation_times'].append(density_calc_time)
        
        density = bucket_density * self.bucket_density_factor
        
        # Update tracking before updating stats (using layer 0 bucket as primary for tracking)
        self._track_bucket_access(buckets[0], density, timestamp, embedding)
        self.tracking_data['neighbor_contributions'].append(neighbor_contribution)
        
        # Update stats for all layers
        self._update_stats_all_layers(buckets)

        curve = self.config.experiment.curve

        if curve == "exponential":
            # Exponential with sensitivity
            sensitivity = self.config.experiment.sensitivity
            temperature = 2.0 * math.exp(-sensitivity * density)
        else:
            # Rational with decay rate
            decay_rate = self.config.experiment.decay_rate
            temperature = 2.0 / (1 + decay_rate * density)
        
        # Track temperature for primary bucket (layer 0)
        self.tracking_data['bucket_temperatures'][buckets[0]].append(temperature)
        self.tracking_data['bucket_densities'][buckets[0]].append(density)

        # Count unique buckets across all layers
        total_unique_buckets = sum(len(layer_counts) for layer_counts in self.bucket_counts)

        # Enhanced debug info with multi-layer details
        debug_info = {
            "lsh_buckets": buckets,  # List of bucket IDs per layer
            "lsh_bucket": buckets[0],  # Primary bucket (layer 0) for backward compatibility
            "bucket_counts": [self.bucket_counts[layer_idx][buckets[layer_idx]] for layer_idx in range(lsh_layers)],
            "per_layer_densities": [round(d, 3) for d in per_layer_densities],
            "bucket_density": round(bucket_density, 3),  # Averaged density
            "neighbor_contribution": round(neighbor_contribution, 3),
            "density": round(density, 3),
            "temperature": round(temperature, 3),
            "bucket_age": self._get_bucket_age(buckets[0]),
            "total_unique_buckets": total_unique_buckets,
            "lsh_layers": lsh_layers,
            "bucket_reuse_rate": self._calculate_bucket_reuse_rate(),
            "hamming_distance_from_last": self.tracking_data['hamming_distances'][-1] if self.tracking_data['hamming_distances'] else 0,
            "computation_time_ms": round((bucket_time + density_calc_time) * 1000, 2)
        }
        
        return temperature, debug_info
    
    def _calculate_bucket_density_for_layer(self, bucket: str, layer_idx: int) -> Tuple[float, float]:
        """Calculate density for a bucket and its neighbors in a specific layer, returning both total density and neighbor contribution"""
        if self.total_count == 0:
            return 0.0, 0.0
        
        layer_bucket_counts = self.bucket_counts[layer_idx]
        main_count = layer_bucket_counts[bucket]
        
        neighbor_count = 0
        for i in range(len(bucket)):
            neighbor = list(bucket)
            neighbor[i] = '1' if bucket[i] == '0' else '0'
            neighbor_key = ''.join(neighbor)
            neighbor_count += layer_bucket_counts.get(neighbor_key, 0)
        
        relevant_count = main_count + neighbor_count * 0.5  # Neighbors weighted less
        
        density = min(relevant_count / (self.total_count * 0.1), 1.0) # clamping density to stay between 1.0 and 0.0. The '*0.1' is used to configure the % of total items in a bucket and its neighbors to trigger maximum density output
        neighbor_contribution = (neighbor_count * 0.5) / max(relevant_count, 1)
        
        return density, neighbor_contribution

    def _calculate_knn_bucket_density_for_layer(self, bucket: str, layer_idx: int) -> Tuple[float, float]:
        if self.total_count == 0:
            return 0.0, 0.0

        layer_bucket_counts = self.bucket_counts[layer_idx]
        main_count = layer_bucket_counts[bucket]

        k_nearest = self._get_k_nearest_neighbors_BFS(bucket, layer_bucket_counts, self.k)    

        # these metrics are not needed but might be useful later on
#        nearest_neighbors = dict([(n, get_hamming_distance(bucket, n)) for n in k_nearest])
#        if nearest_neighbors:
#            closest_bucket, closest_distance = min(nearest_neighbors.items(), key=lambda item: item[1])
#        else: 
#            closest_bucket, closest_distance = None, 0.0
#        avg_distance = sum(nearest_neighbors.values()) / len(k_nearest)

        weighted_neighbor_count = 0
        for neighbor in k_nearest:
            dist = get_hamming_distance(bucket, neighbor)
            count = layer_bucket_counts.get(neighbor, 0)
            weight = 1.0 / (dist+1)
            weighted_neighbor_count += count * weight
        
        relevant_count = main_count + weighted_neighbor_count
        density = min(relevant_count/(self.total_count*0.1), 1.0)
        neighbor_contribution = weighted_neighbor_count/max(relevant_count, 1)
        
        return density, neighbor_contribution

    
    def _calculate_bucket_density_layered(self, buckets: List[str]) -> Tuple[float, float, List[float]]:
        """
        Calculate density averaged across all layers.
        
        Args:
            buckets: List of bucket IDs, one per layer
            
        Returns:
            (average_density, average_neighbor_contribution, per_layer_densities)
        """
        if self.total_count == 0:
            return 0.0, 0.0, [0.0] * len(buckets)
        
        layer_densities = []
        layer_neighbor_contributions = []
        
        for layer_idx, bucket in enumerate(buckets):
            if self.knn_density:
                density, neighbor_contrib = self._calculate_knn_bucket_density_for_layer(bucket, layer_idx)
            else:
                density, neighbor_contrib = self._calculate_bucket_density_for_layer(bucket, layer_idx)
            layer_densities.append(density)
            layer_neighbor_contributions.append(neighbor_contrib)
        
        avg_density = sum(layer_densities) / len(layer_densities)
        avg_neighbor_contribution = sum(layer_neighbor_contributions) / len(layer_neighbor_contributions)
        
        return avg_density, avg_neighbor_contribution, layer_densities
    
    def _get_neighbor_key(self, bucket, i):
        neighbor = list(bucket)
        neighbor[i] = '1' if bucket[i] == '0' else '0'
        neighbor_key = ''.join(neighbor)
        return neighbor_key
    
    def _get_neighbors(self, bucket):
        neighbors = []
        for i in range(len(bucket)):
            neighbor = self._get_neighbor_key(bucket, i)
            neighbors.append(neighbor)
        return neighbors

    # This is a modified breadth-first-search
    def _get_k_nearest_neighbors_BFS(self, bucket, buckets, k):
        q = deque()
        k_nearest = []
        # dict for tracking already seen buckets
        seen_buckets = {}
        # label root as explored
        seen_buckets[bucket] = True
        q.append(bucket)

        while q and k > 0:
            curr = q.popleft()
            for n in self._get_neighbors(curr):
                if k <= 0:
                    break
                if n not in seen_buckets:
                    seen_buckets[n] = True
                    if buckets.get(n, 0) > 0:
                        k_nearest.append(n)
                        k -= 1
                    q.append(n)

        return k_nearest

    def _average_neighbor_distance(self, bucket: str, layer_idx: int, k: int) -> Tuple[float, str, float]:
        """Calculate the average distance d of the k nearest neighbors, returning both d and the nearest neighbor n"""
        if self.total_count == 0:
            return 0.0, "", 0.0

        layer_bucket_counts = self.bucket_counts[layer_idx]
        k_nearest = self._get_k_nearest_neighbors_BFS(bucket, layer_bucket_counts, k)
        nearest_neighbors = dict([(n, get_hamming_distance(bucket, n)) for n in k_nearest])

        if nearest_neighbors:
            closest_bucket, closest_distance = min(nearest_neighbors.items(), key=lambda item: item[1])
        else: 
            closest_bucket, closest_distance = None, 0.0

        if not k_nearest:
            return 0.0, None, 0.0
        
        avg_distance = sum(nearest_neighbors.values()) / len(k_nearest)

        return avg_distance, closest_bucket, closest_distance

    
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
        """Calculate what percentage of queries reuse existing buckets (averaged across layers)"""
        if self.total_count == 0:
            return 0.0
        # Count unique buckets across all layers
        total_unique_buckets = sum(len(layer_counts) for layer_counts in self.bucket_counts)
        avg_unique_per_layer = total_unique_buckets / self.lsh_layers
        reuse_rate = 1.0 - (avg_unique_per_layer / self.total_count)
        return round(max(0.0, reuse_rate), 3)
    
    def _update_stats_for_layer(self, bucket: str, layer_idx: int):
        """Update statistics for a specific layer"""
        self.bucket_history[layer_idx].append(bucket)
        self.bucket_counts[layer_idx][bucket] += 1
    
    def _update_stats_all_layers(self, buckets: List[str]):
        """Update statistics for all layers and increment total_count once"""
        for layer_idx, bucket in enumerate(buckets):
            self._update_stats_for_layer(bucket, layer_idx)
        self.total_count += 1
        
        # Update performance metrics
        self.performance_metrics['bucket_reuse_rate'] = self._calculate_bucket_reuse_rate()
    
    def register_item(self, item_id: int, embedding: np.ndarray = None, buckets: List[str] = None) -> List[str]:
        """
        Register a cache item with its LSH buckets (one per layer).
        Called when a new item is added to the cache.
        
        Args:
            item_id: The unique ID of the cached item (from scalar storage)
            embedding: The embedding vector for this item (optional if buckets provided)
            buckets: Pre-computed bucket IDs per layer (optional, avoids recomputation)
            
        Returns:
            The list of bucket IDs this item was assigned to (one per layer)
        """
        # Use provided buckets, or cached buckets, or compute them
        if buckets is None:
            if embedding is not None:
                # Check if we can reuse the cached buckets
                embedding_hash = hash(embedding.tobytes()) if hasattr(embedding, 'tobytes') else hash(tuple(embedding))
                if self.last_buckets is not None and self.last_bucket_embedding_hash == embedding_hash:
                    buckets = self.last_buckets
                else:
                    buckets, _ = self.get_layered_lsh_bucket(embedding)
            elif self.last_buckets is not None:
                # Fall back to last cached buckets
                buckets = self.last_buckets
            else:
                raise ValueError("Either embedding or buckets must be provided")

        print("Item registered!")
        
        # Store item_id → list of buckets (one per layer)
        self.item_to_bucket[item_id] = buckets
        
        # Add item to each layer's bucket_to_items mapping
        for layer_idx, bucket in enumerate(buckets):
            self.bucket_to_items[layer_idx][bucket].add(item_id)
        
        self.active_item_count += 1
        
        return buckets
    
    def evict_item(self, item_id: int) -> bool:
        """
        Handle eviction of a cache item - decrement bucket counts in all layers.
        Called when an item is evicted from the cache.
        
        Args:
            item_id: The unique ID of the evicted item
            
        Returns:
            True if item was found and evicted, False otherwise
        """
        # TODO: we need to make sure that self.bucket_count entries get removed once their value drops back to 0, otherwise they might skew neighborhood calculations
        buckets = self.item_to_bucket.get(item_id)
        
        print("evicting item")

        if buckets is None:
            print("fail")
            # Item wasn't registered (might be from before LSH tracking)
            return False
        
        # Decrement bucket count in each layer
        for layer_idx, bucket in enumerate(buckets):
            if self.bucket_counts[layer_idx][bucket] > 0:
                self.bucket_counts[layer_idx][bucket] -= 1
            
            # Remove from tracking structures
            self.bucket_to_items[layer_idx][bucket].discard(item_id)
            
            # Clean up empty bucket sets
            if len(self.bucket_to_items[layer_idx][bucket]) == 0:
                del self.bucket_to_items[layer_idx][bucket]
        
        # Update total count (once, not per layer)
        if self.total_count > 0:
            self.total_count -= 1
        
        del self.item_to_bucket[item_id]
        self.active_item_count -= 1
        
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
        """Get statistics about active (non-evicted) items per bucket (across all layers)."""
        # Aggregate bucket stats across all layers
        buckets_with_active_items = sum(len(layer_bucket_to_items) for layer_bucket_to_items in self.bucket_to_items)
        items_per_bucket_per_layer = [
            {b: len(items) for b, items in layer_bucket_to_items.items()}
            for layer_bucket_to_items in self.bucket_to_items
        ]
        return {
            'total_active_items': self.active_item_count,
            'total_tracked_items': len(self.item_to_bucket),
            'buckets_with_active_items': buckets_with_active_items,
            'items_per_bucket_per_layer': items_per_bucket_per_layer,
        }
    
    def save_tracking_data(self, filepath: str):
        """Save tracking data to JSON file for analysis"""
        # Aggregate bucket counts across all layers
        all_bucket_counts = {}
        for layer_idx, layer_counts in enumerate(self.bucket_counts):
            for bucket, count in layer_counts.items():
                all_bucket_counts[f"L{layer_idx}_{bucket}"] = count
        
        total_unique_buckets = sum(len(layer_counts) for layer_counts in self.bucket_counts)
        
        export_data = {
            'configuration': {
                'num_hyperplanes': self.num_hyperplanes,
                'embedding_dim': len(self.hyperplanes[0]),
                'lsh_layers': self.lsh_layers,
                'total_queries': self.total_count,
                'unique_buckets': total_unique_buckets,
            },
            'bucket_statistics': {
                'bucket_counts': all_bucket_counts,
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
        """Get comprehensive analytics about bucket usage (aggregated across layers)"""
        # Check if any layer has bucket counts
        if not any(self.bucket_counts):
            return {}
        
        # Aggregate bucket access counts across all layers
        all_bucket_access_counts = []
        for layer_counts in self.bucket_counts:
            all_bucket_access_counts.extend(layer_counts.values())
        
        if not all_bucket_access_counts:
            return {}
        
        bucket_lifetimes = []
        # Use tracking_data which tracks layer 0 buckets
        for bucket in self.bucket_counts[0]:
            if bucket in self.tracking_data['bucket_first_seen'] and bucket in self.tracking_data['bucket_last_seen']:
                lifetime = (self.tracking_data['bucket_last_seen'][bucket] - 
                           self.tracking_data['bucket_first_seen'][bucket]).total_seconds()
                bucket_lifetimes.append(lifetime)
        
        total_unique_buckets = sum(len(layer_counts) for layer_counts in self.bucket_counts)
        
        # Find most accessed buckets across all layers
        all_buckets_with_layer = []
        for layer_idx, layer_counts in enumerate(self.bucket_counts):
            for bucket, count in layer_counts.items():
                all_buckets_with_layer.append((f"L{layer_idx}_{bucket}", count))
        most_accessed = sorted(all_buckets_with_layer, key=lambda x: x[1], reverse=True)[:10]
        
        analytics = {
            'total_buckets': total_unique_buckets,
            'lsh_layers': self.lsh_layers,
            'total_queries': self.total_count,
            'bucket_reuse_rate': self._calculate_bucket_reuse_rate(),
            'most_accessed_buckets': most_accessed,
            'access_distribution': {
                'mean': np.mean(all_bucket_access_counts),
                'median': np.median(all_bucket_access_counts),
                'std': np.std(all_bucket_access_counts),
                'max': max(all_bucket_access_counts),
                'min': min(all_bucket_access_counts),
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
    
    def register_item(self, item_id: int, embedding: np.ndarray = None, buckets: List[str] = None) -> List[str]:
        """
        Register a cached item with its LSH buckets (one per layer).
        
        Args:
            item_id: The unique ID of the cached item
            embedding: The embedding vector (optional if buckets provided or cached)
            buckets: Pre-computed bucket IDs from debug_info['lsh_buckets'] (optional)
        """
        return self.estimator.register_item(item_id, embedding, buckets)
    
    def evict_item(self, item_id: int) -> bool:
        """Handle eviction of a single item."""
        return self.estimator.evict_item(item_id)
    
    def evict_items(self, item_ids: List[int]) -> int:
        """Handle batch eviction of items."""
        return self.estimator.evict_items(item_ids)
    
    def get_active_stats(self) -> Dict[str, Any]:
        """Get active item statistics."""
        return self.estimator.get_active_bucket_stats()