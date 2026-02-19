from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_dir: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    image_size: int
    batch_size: int
    num_workers: int
    pin_memory: bool
