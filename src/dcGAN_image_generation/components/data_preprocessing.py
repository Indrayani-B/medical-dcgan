import os
import json
from pathlib import Path
from dcGAN_image_generation import logger
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dcGAN_image_generation.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def _get_transforms(self):
        logger.info("Creating DCGAN-compatible transforms")

        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5])

        ])

        return transform

    def _load_dataset(self):
        logger.info("Loading dataset from ingestion artifacts")

        dataset = datasets.ImageFolder(
            root=self.config.data_path / "train",
            transform=self._get_transforms()
        )

        logger.info(f"Dataset loaded with {len(dataset)} images")

        return dataset

    def save_metadata(self) -> None:
        """
        Saves dataset metadata for reproducibility and pipeline tracking
        """

        try:
            dataset = self._load_dataset()

            metadata = {
                "dataset_size": len(dataset),
                "image_size": self.config.image_size,
                "batch_size": self.config.batch_size
            }

            metadata_path = Path(self.config.root_dir) / "metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"Metadata saved at {metadata_path}")

        except Exception as e:
            raise e
        
    def validate_batch(self) -> None:
        """
        Validates one batch for:
        - shape
        - pixel range
        """
    
        try:
            dataset = self._load_dataset()
    
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,  # safe for Windows
                pin_memory=self.config.pin_memory
            )
    
            images, _ = next(iter(dataloader))
    
            logger.info(f"Batch shape: {images.shape}")
            logger.info(f"Pixel min: {images.min().item()}")
            logger.info(f"Pixel max: {images.max().item()}")
    
            if images.shape[1:] != (3, self.config.image_size, self.config.image_size):
                raise ValueError("Image shape mismatch for DCGAN")
    
            if images.min() < -1.1 or images.max() > 1.1:
                raise ValueError("Pixel range not in [-1, 1]")
    
            logger.info("Batch validation successful")
    
        except Exception as e:
            raise e
    