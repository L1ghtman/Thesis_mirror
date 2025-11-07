import json
import yaml
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class SystemConfig:
    info_level: int
    embedding_model: str

@dataclass
class LoggingConfig:
    level: str
    format: str

@dataclass
class VectorStoreConfig:
    dimension: int
    index_type: str
    metric_type: int

@dataclass
class CacheConfig:
    CACHE_DIR: str

@dataclass
class ExperimentConfig:
    name: str
    run_id: str
    dataset_name: str
    load_from_file: bool
    sample_size: int
    partial_questions: bool
    range_min: int
    range_max: int
    use_cache: bool
    use_temperature: bool
    max_cache_size: int
    cache_strategy: str
    use_LSH: bool
    bucket_density_factor: float
    num_hyperplanes: int
    window_size: int
    sensitivity: float
    decay_rate: float

@dataclass
class Config:
    sys: SystemConfig
    logging: LoggingConfig
    vector_store: VectorStoreConfig
    cache: CacheConfig
    experiment: ExperimentConfig

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            return cls(**yaml.safe_load(f))
        
# global instance
_config: Optional[Config] = None

def load_config(path: str) -> Config:
    """Load config from yaml and set global instance"""
    global _config
    _config = Config.from_yaml(path)
    return _config

def get_config() -> Config:
    """Get global config instance"""
    if _config is None:
        raise RuntimeError("Config not loaded, use load_config() first.")
    return _config