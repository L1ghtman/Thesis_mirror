import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

@dataclass
class SystemConfig:
    info_level: int

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
    dataset: str
    max_cache_size: int
    cache_strategy: str
    partial_questions: bool
    use_LSH: bool
    use_cache: bool

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
        