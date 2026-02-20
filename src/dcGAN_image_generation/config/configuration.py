from dcGAN_image_generation.constants import *
from dcGAN_image_generation.utils.common import read_yaml, create_directories
from dcGAN_image_generation.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig, ModelTrainingConfig


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
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        config = self.config.data_preprocessing
    
        create_directories([config.root_dir])
    
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
    
        return data_preprocessing_config    
    
    def get_model_training_config(self) -> ModelTrainingConfig:
    
        config = self.config.model_training
        params = self.params.model_training
        preprocessing_config = self.config.data_preprocessing
    
        create_directories([
            config.root_dir,
            config.trained_model_dir,
            config.generated_images_dir
        ])
        
        return ModelTrainingConfig(
            root_dir=ROOT_DIR / config.root_dir,
            data_path=ROOT_DIR / preprocessing_config.data_path,
    
            trained_model_dir=ROOT_DIR / config.trained_model_dir,
            generated_images_dir=ROOT_DIR / config.generated_images_dir,
    
            z_dim=params.z_dim,
            generator_feature_maps=params.generator_feature_maps,
            discriminator_feature_maps=params.discriminator_feature_maps,
    
            num_epochs=params.num_epochs,
            learning_rate=params.learning_rate,
            beta1=params.beta1,
            save_image_interval=params.save_image_interval,
            image_size=preprocessing_config.image_size,

            batch_size=preprocessing_config.batch_size,
            
            mlflow_uri=config.mlflow_uri,

            all_params={
                "z_dim": params.z_dim,
                "gen_feature_maps": params.generator_feature_maps,
                "disc_feature_maps": params.discriminator_feature_maps,
                "learning_rate": params.learning_rate,
                "beta1": params.beta1,
                "epochs": params.num_epochs,
                "batch_size": preprocessing_config.batch_size,
                "image_size": preprocessing_config.image_size
            }
        )

    