from dcGAN_image_generation import logger
from dcGAN_image_generation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from dcGAN_image_generation.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from dcGAN_image_generation.pipeline.stage_03_model_training import ModelTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Preprocessing Stage"
try:
    logger.info(f"*******************************")    
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = DataPreprocessingPipeline()
    obj.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Model Training Stage"
try:
        logger.info(f"*******************************")    
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e