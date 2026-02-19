import os
import shutil
from dcGAN_image_generation.entity.config_entity import DataIngestionConfig
from dcGAN_image_generation import logger
from dcGAN_image_generation.utils.common import get_size
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def copy_data(self) -> None:
        """
        Copies dataset from source_dir → artifacts ingestion folder
        """

        try:
            source = Path(self.config.source_dir)
            destination = Path(self.config.unzip_dir)

            logger.info(f"Source directory: {source}")
            logger.info(f"Destination directory: {destination}")

            if not source.exists():
                raise FileNotFoundError(f"Source data not found at {source}")

            if destination.exists():
                logger.info("Data already ingested. Skipping copy.")
                return

            shutil.copytree(src=source, dst=destination)

            logger.info(
                f"Data successfully ingested from {source} to {destination}"
            )

            logger.info(f"Ingested data size: {get_size(destination)}")

        except Exception as e:
            raise e
