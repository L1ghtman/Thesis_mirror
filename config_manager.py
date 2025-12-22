import json
import yaml
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class SystemConfig:
    info_level: int
    model: str
    system_prompt: str
    embedding_model: str
    hpc: bool
    url: str
    temperature: float              = 0.1

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
    name: str                       = ""
    run_id: str                     = ""
    dataset_name: str               = ""
    load_from_file: bool            = False
    sample_size: int                = 100
    partial_questions: bool         = True
    range_min: int                  = 0
    range_max: int                  = 100
    use_cache: bool                 = True
    use_temperature: bool           = True
    max_cache_size: int             = 1000
    cache_strategy: str             = "memory"
    eviction_policy: str            = "LFU"
    use_LSH: bool                   = False
    bucket_density_factor: float    = 1.0
    num_hyperplanes: int            = 8
    window_size: int                = 2000
    curve: str                      = "rational"
    sensitivity: float              = 2.0
    decay_rate: float               = 5.0

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
            data = yaml.safe_load(f)
        return cls(
            sys=SystemConfig(**data['sys']),
            logging=LoggingConfig(**data['logging']),
            vector_store=VectorStoreConfig(**data['vector_store']),
            cache=CacheConfig(**data['cache']),
            experiment=ExperimentConfig(**data['experiment']),
        )
        
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