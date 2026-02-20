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
 

@dataclass(frozen=True)
class ModelTrainingConfig:

    root_dir: Path
    data_path: Path

    trained_model_dir: Path
    generated_images_dir: Path

    z_dim: int
    generator_feature_maps: int
    discriminator_feature_maps: int

    num_epochs: int
    learning_rate: float
    beta1: float
    save_image_interval: int

    batch_size: int
    image_size: int
    
    mlflow_uri: str
    all_params: dict


