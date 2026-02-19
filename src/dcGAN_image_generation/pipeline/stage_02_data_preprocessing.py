from dcGAN_image_generation.config.configuration import ConfigurationManager
from dcGAN_image_generation.components.data_preprocessing import DataPreprocessing
from dcGAN_image_generation import logger

STAGE_NAME = "Data Preprocessing Stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.save_metadata()
        data_preprocessing.validate_batch()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e