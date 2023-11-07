from fraud.config.Configuration import Configuration 
from fraud.logger import logging
from fraud.exception import FraudException

from collections import namedtuple
from datetime import datetime
import uuid
from fraud.config.Configuration import Configuration
from fraud.logger import logging

from threading import Thread
from typing import List
from multiprocessing import Process

from fraud.entity.artifact_entity import *
from fraud.entity.config_entity import DataIngestionConfig,ModelEvaluationConfig
from fraud.components.data_ingestion import DataIngestion
from fraud.components.data_validation import DataValidation
from fraud.components.data_transformation import DataTransformation
from fraud.components.model_trainer import ModelTrainer
from fraud.components.model_evaluation import ModelEvaluation
from fraud.components.model_pusher import ModelPusher
import os,sys

from collections import namedtuple
from datetime import datetime
import pandas as pd
from fraud.constant import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME




Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                       "running_status", "start_time", "stop_time", "execution_time", "message",
                                       "experiment_file_path", "accuracy", "is_model_accepted"])



class Pipeline(Thread):
    config = Configuration()
    experiment: Experiment = Experiment(*([None] * 11))
    experiment_file_path = os.path.join(config.training_pipeline_config.artifact_dir,
                                        EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)

    def __init__(self,config: Configuration = Configuration()) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            super().__init__(daemon=False, name="pipeline")
            self.config = config
        except Exception as e:
            raise FraudException(e,sys) from e

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise FraudException(e,sys) from e    


    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact) \
        -> DataValidationArtifact :
        try:
            data_validation =  DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                              data_ingestion_artifact=data_ingestion_artifact
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise FraudException(e,sys) from e

    def start_data_transformation(self,
                                data_ingestion_artifact:DataIngestionArtifact,
                                data_validation_artifact: DataValidationArtifact
                                )->DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise FraudException(e,sys)

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(),
                                         data_transformation_artifact=data_transformation_artifact
                                         ) #Object of ModelTrainer class in the components section
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise FraudException(e, sys) from e

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact, model_trainer_config = self.config.get_model_trainer_config())
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise FraudException(e, sys) from e

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.config.get_model_pusher_config(),
                model_evaluation_artifact=model_eval_artifact
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise FraudException(e, sys) from e

    def run_pipeline(self):
        try:

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)

            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info("Trained model rejected.")
            logging.info("Pipeline completed.")

        except Exception as e:
            raise FraudException(e, sys) from e



        