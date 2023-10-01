from fraud.entity.config_entity import DataIngestionConfig
from fraud.entity.artifact_entity import DataIngestionArtifact
from fraud.constant import *
from fraud.config .Configuration import Configuration
from fraud.exception import FraudException
from fraud.logger import logging
from fraud.components.db_operation import MongoDB

import os
import sys
import tarfile
import numpy as np
import pandas as pd
import shutil

from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise FraudException(e,sys)
    def download_data(self,) -> str:
        try:
            # remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            #folder location to download file
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            #if os.path.exists(raw_data_dir):
                #os.remove(raw_data_dir)
            
            os.makedirs(raw_data_dir,exist_ok=True)

            banking_file_name = FILE_NAME

            raw_data_file_path = os.path.join(raw_data_dir, banking_file_name)

            logging.info(f"Downloading file from :[{download_url}] into :[{raw_data_file_path}]")
            urllib.request.urlretrieve(download_url, raw_data_file_path)
            logging.info(f"File :[{raw_data_file_path}] has been downloaded successfully.")
            return raw_data_file_path

        except Exception as e:
            raise FraudException(e,sys) from e
    def get_data(self) -> str:
        try:
            # Folder location to retrieve the file
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            os.makedirs(raw_data_dir,exist_ok=True)
            logging.info(" Raw data Directory ...")
            
            banking_file_name = FILE_NAME
            raw_data_file_path = os.path.join(raw_data_dir, banking_file_name)
            
            # Local location of dataset
            dataset_location = self.data_ingestion_config.dataset_location

            # Copy the file from dataset location to raw data file
            shutil.copy(dataset_location, raw_data_file_path)
            logging.info(f"File copied from {dataset_location} to {raw_data_file_path}")

            return raw_data_file_path

        except Exception as e:
            raise FraudException(str(e)) from e
        
        
    def split_data_as_train_test(self,raw_data_file_path) -> DataIngestionArtifact:
        try:


            banking_file_path = raw_data_file_path


            logging.info(f"Reading csv file: [{banking_file_path}]")
            banking_data_frame = pd.read_csv(banking_file_path)
            
            """
            NOTE: We are creating subsamples and this will lead to DATA LEAKAGE, but as the original dataset is huge and
            there is limitation of my local sysytem i am resampling considering all the best practices
            """
            self.db.create_and_check_collection()

            data_dict = banking_data_frame.to_dict("records")
            logging.info(f"Inserting file:  into DB")
            self.db.insertall(data_dict)

            # fetching the data set from DB
            logging.info(f"Fetching entire data from DB")
            dataframe = self.db.fetch_df()
            dataframe.drop(columns = "_id",inplace=True)
            logging.info(f"Entire data fetched successfully from DB!!!")

            
            fraud_df = dataframe.loc[banking_data_frame['isFraud'] == 1]
            non_fraud_df = dataframe[banking_data_frame['isFraud'] == 0][:len(fraud_df)]
            banking_data_frame= pd.concat([fraud_df, non_fraud_df])
            # reset index
            banking_data_frame.reset_index(drop=True, inplace=True)
            #to keep the distribution equivalent in train and test dataset we make a new category on which we split using strat. shuffle split
            
            banking_data_frame["cat_amount"] = pd.cut(
                banking_data_frame["amount"], 
                4,
                labels=[1,2,3,4]
            )                          

            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index,test_index in sss.split(banking_data_frame, banking_data_frame["cat_amount"]):
                strat_train_set = banking_data_frame.loc[train_index].drop(["cat_amount"],axis=1)
                strat_test_set = banking_data_frame.loc[test_index].drop(["cat_amount"],axis=1)
            
            logging.info("Inserting new Training Data into DB")
            self.db.create_and_check_collection(coll_name="Training")
            self.db.insertall(strat_train_set.to_dict("records"))

            logging.info("Inserting new Test Data into DB")
            self.db.create_and_check_collection(coll_name="Test")
            self.db.insertall(strat_test_set.to_dict("records"))

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            'train.csv')

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        'test.csb')
            
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)
            

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise FraudException(e,sys) from e
        
        
        
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            raw_data_file_path=self.get_housing_data()
         #   raw_data_file_path =  self.download_housing_data()
            return self.split_data_as_train_test(raw_data_file_path)
        except Exception as e:
            raise FraudException(e,sys) from e
        
    