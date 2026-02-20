import os
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
from dcGAN_image_generation.config.configuration import ConfigurationManager
from dcGAN_image_generation.components.model_trainig import ModelTraining
from dcGAN_image_generation import logger
 
STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        
        # Load config
        # ================================
        config = ConfigurationManager()
        training_config = config.get_model_training_config()

        # ================================
        # DagsHub authentication
        # ================================
        os.environ["MLFLOW_TRACKING_USERNAME"] =  os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

        mlflow.set_tracking_uri(training_config.mlflow_uri)
        mlflow.set_experiment("DCGAN_Chest_Xray")

        # ================================
        # Start MLflow run
        # ================================
        with mlflow.start_run():

            # 🔹 log all hyperparameters
            mlflow.log_params(training_config.all_params)

            # ================================
            # Training
            # ================================
            trainer = ModelTraining(config=training_config)

            trainer.get_dataloader()
            trainer.build_generator()
            trainer.build_discriminator()

            final_G_loss, final_D_loss = trainer.train()

            # ================================
            # Log metrics
            # ================================
            mlflow.log_metric("G_loss", final_G_loss)
            mlflow.log_metric("D_loss", final_D_loss)

            # ================================
            # Log generated images
            # ================================
            mlflow.log_artifacts(
                str(training_config.generated_images_dir),
                artifact_path="generated_images"
            )

            # ================================
            # Log models
            # ================================
            mlflow.pytorch.log_model(trainer.netG, "generator")
            mlflow.pytorch.log_model(trainer.netD, "discriminator")

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
