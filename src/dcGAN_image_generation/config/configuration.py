from dcGAN_image_generation.constants import *
from dcGAN_image_generation.utils.common import read_yaml, create_directories
from dcGAN_image_generation.entity.config_entity import DataIngestionConfig


class ConfigurationManager:

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        self.artifacts_root = ROOT_DIR / self.config.artifacts_root
        create_directories([self.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        root_dir = ROOT_DIR / config.root_dir

        create_directories([root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=root_dir,
            source_dir=ROOT_DIR / config.source_dir,
            local_data_file=ROOT_DIR / config.local_data_file,
            unzip_dir=ROOT_DIR / config.unzip_dir
        )

        return data_ingestion_config