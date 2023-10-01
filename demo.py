from fraud.pipeline.pipeline import Pipeline
from fraud.exception import FraudException
from fraud.logger import logging
from fraud.config.Configuration import Configuration
from fraud.components.data_transformation import DataTransformation
import os


def main():
    try:
        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuration(config_file_path=config_path))
        #pipeline.run_pipeline()
        pipeline.run_pipeline()
        logging.info("main function execution completed.")


    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()