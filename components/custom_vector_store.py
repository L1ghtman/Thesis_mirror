import faiss
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from gptcache.manager.vector_data.base import VectorBase, VectorData

class FAISSVectorStore(VectorBase):
    def __init__(self, dimension: int, index_type: str = "Flat"):
        """
        Initialize FAISS vector store
        Args:
            dimension: Vector dimension
            index_type: FAISS index type ("Flat", "IVF", "HNSW", etc.)
        """
        self.dimension = dimension
        self.id_to_index: Dict[int, int] = {}  # Maps external IDs to FAISS indices
        self.index_to_id: Dict[int, int] = {}  # Maps FAISS indices to external IDs
        self.next_index = 0
        
        # Initialize FAISS index
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            # IVF index with 100 centroids
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.train(np.zeros((100, dimension), dtype=np.float32))
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def mul_add(self, datas: List[VectorData]):
        """Add multiple vectors to the index"""
        if not datas:
            return
            
        # Prepare vectors and map IDs
        vectors = []
        for data in datas:
            vectors.append(data.data.astype(np.float32))
            self.id_to_index[data.id] = self.next_index
            self.index_to_id[self.next_index] = data.id
            self.next_index += 1
            
        # Add to FAISS index
        vectors_array = np.stack(vectors)
        self.index.add(vectors_array)

    def search(self, data: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """
        Search for similar vectors
        Returns: List of (id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
            
        # Prepare query vector
        query = data.astype(np.float32).reshape(1, -1)
        
        # Search in FAISS index
        distances, indices = self.index.search(query, top_k)
        
        # Convert to list of (id, similarity) tuples
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS returns -1 for padding when there are fewer results
                original_id = self.index_to_id[idx]
                # Convert distance to similarity score (1 / (1 + distance))
                similarity = 1 / (1 + dist)
                results.append((original_id, float(similarity)))
                
        return results

    def rebuild(self, ids=None) -> bool:
        """Rebuild index for specified ids or all vectors"""
        if ids is None:
            return True
            
        # Get vectors for specified ids
        vectors = []
        new_id_to_index = {}
        new_index_to_id = {}
        next_index = 0
        
        for id in ids:
            if id in self.id_to_index:
                idx = self.id_to_index[id]
                vector = self.get_embeddings(id)
                if vector is not None:
                    vectors.append(vector)
                    new_id_to_index[id] = next_index
                    new_index_to_id[next_index] = id
                    next_index += 1
                    
        # Reset index
        self.index.reset()
        
        # Add vectors back
        if vectors:
            vectors_array = np.stack(vectors)
            self.index.add(vectors_array)
            
        self.id_to_index = new_id_to_index
        self.index_to_id = new_index_to_id
        self.next_index = next_index
        
        return True

    def delete(self, ids) -> bool:
        """Delete vectors (Note: FAISS doesn't support direct deletion)"""
        if isinstance(ids, (int, str)):
            ids = [ids]
            
        # Remove from ID mappings
        for id in ids:
            if id in self.id_to_index:
                idx = self.id_to_index[id]
                del self.id_to_index[id]
                del self.index_to_id[idx]
                
        # Rebuild index without deleted vectors
        remaining_ids = list(self.id_to_index.keys())
        return self.rebuild(remaining_ids)

    def get_embeddings(self, data_id: Union[int, str]) -> Optional[np.ndarray]:
        """Get vector by id"""
        idx = self.id_to_index.get(int(data_id))
        if idx is None:
            return None
            
        # Reconstruct vector from index
        if hasattr(self.index, 'reconstruct'):
            return self.index.reconstruct(idx)
        return None

    def update_embeddings(self, data_id: Union[int, str], emb: np.ndarray):
        """Update vector (requires delete and re-add in FAISS)"""
        self.delete(data_id)
        self.mul_add([VectorData(id=int(data_id), data=emb)])

    def close(self):
        """Cleanup resources"""
        if hasattr(self.index, 'reset'):
            self.index.reset()
        self.id_to_index.clear()
        self.index_to_id.clear()