import time
import logging
from components import cache_analyzer
import sys
import traceback
from gptcache.core import Cache
from gptcache.embedding import SBERT
import numpy as np
from transformers import AutoTokenizer, AutoModel

def embedding_func(prompt, extra_param=None):
    encoder = SBERT('all-MiniLM-L6-v2')
    embedding = encoder.to_embeddings(prompt)
    return tuple(embedding)

def non_normalized_embedding_func(prompt, extra_param=None):
    encoder = SBERT('all-MiniLM-L6-v2')
    embedding = encoder.to_embeddings(prompt, normalize_embeddings=True)
    return tuple(embedding)

def temperature_func(clusterer, embedding_data, cache_size, temperature):
    """
    Calculate the temperature based on the clustering and cache size.
    Returns a tuple of (temperature, cluster_id) if clustering successful,
    otherwise just returns the temperature.
    """
    # Create a unique ID for this vector using a hash of the embedding
    vector_id = hash(tuple(embedding_data))
    
    # Add to buffer with ID
    clusterer.add_to_buffer(embedding_data, vector_id)
    
    # Process buffer, force processing even if below min size
    processed = False
    try:
        if len(clusterer.vectors_buffer) >= 1:  # Even with just one vector, try to process
            processed = clusterer.process_buffer(min_batch_size=1)
    except Exception as e:
        print(f"Buffer processing error (normal): {e}")
    
    # If normal processing failed or wasn't attempted, force it
    if not processed:
        try:
            processed_count = clusterer.force_process_buffer()
            if processed_count > 0:
                print(f"Forced processing of {processed_count} vectors in buffer")
            else:
                print("No vectors processed")
        except Exception as e:
            print(f"Forced buffer processing error: {e}")
    
    # Get cluster stats and print them
    stats = clusterer.get_cluster_stats()
    if stats:
        print("\n----- CLUSTER STATS -----")
        print(f"Total clusters: {stats.get('num_clusters', 0)}")
        print(f"Total data points: {stats.get('total_points', 0)}")
        print(f"Total vectors seen: {stats.get('total_vectors_seen', stats.get('total_points', 0))}")
        print(f"Non-empty clusters: {len(stats.get('clusters_with_data', []))}/{stats.get('num_clusters', 0)}")
        print(f"Average points per non-empty cluster: {stats.get('average_points_per_cluster', 0):.2f}")
        print("Cluster sizes:")
        for cluster_id, size in sorted(stats.get('cluster_sizes', {}).items()):
            if size > 0:
                print(f"  Cluster {cluster_id}: {size} points")
        print("-------------------------\n")
    
    # Calculate temperature adjustment
    cache_factor = 1.0/max(1, np.log2(cache_size+1))
    
    # Default values
    cluster_adjustment = 1.0
    cluster_id = None
    
    try:
        # Calculate clustering-based adjustment
        cluster_adjustment = clusterer.get_temperature_adjustment(embedding_data)
        
        # Get cluster assignment for this vector if possible
        if clusterer.is_fitted and hasattr(clusterer, 'kmeans') and clusterer.kmeans is not None:
            query_embedding = np.array(embedding_data).reshape(1, -1)
            try:
                cluster_id = int(clusterer.kmeans.predict(query_embedding)[0])
                cluster_size = 0
                if stats and 'cluster_sizes' in stats:
                    cluster_size = stats['cluster_sizes'].get(cluster_id, 0)
                print(f"Current vector assigned to cluster {cluster_id} (size: {cluster_size})")
            except Exception as e:
                print(f"Error getting cluster assignment: {e}")
    except Exception as e:
        print(f"Error in cluster calculations: {e}")
        cluster_adjustment = 1.0  # Default to high temperature on error
    
    # Adjust temperature
    original_temperature = temperature
    adjusted_temperature = original_temperature * (0.4*cache_factor + 0.6*cluster_adjustment)
    temperature = max(0.0, min(2.0, adjusted_temperature))
    
    print(f"Temp adjustment: {original_temperature:.2f} → {temperature:.2f} " +
          f"(cluster: {cluster_adjustment:.2f}, cache: {cache_factor:.2f})")
    print(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .")
    
    # Return both temperature and cluster_id if we have it
    if cluster_id is not None:
        return temperature, cluster_id
    else:
        return temperature

def magnitude_temperature_func(magnitude_cache, embedding):
    """
    Calculate the temperature based on embedding magnitude.
    """
    return magnitude_cache.get_temperature(embedding) 

def lsh_temperature_func(lsh_cache, embedding):
    """
    Calculate the temperature based on LSH bucket density.
    """
    return lsh_cache.get_temperature(embedding)

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

