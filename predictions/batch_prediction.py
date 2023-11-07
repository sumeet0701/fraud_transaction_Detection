from cgi import test
from sklearn import preprocessing
from fraud.exception import FraudException
from fraud.logger import logging


import sys,os
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from fraud.constant import *
from fraud.utils.utils import read_yaml_file,save_object,save_numpy_array_data,load_data,load_numpy_array_data,load_object



class Batch_prediction:

    def __init__(self, file_path
                 ):
        try:
            logging.info(f"{'>' * 20}Data Batch prediction log started.{'<' * 20} ")
            
            self.csv_file_path=file_path
            ROOT_DIR=os.getcwd()
            self.schema_file_path = os.path.join(ROOT_DIR,'config','schema.yaml')
         

        except Exception as e:
            raise BankingException(e,sys) from e

    

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path =self.schema_file_path 

            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            target_column_name = dataset_schema[TARGET_COLUMN_KEY]
            """FROM EDA WE FOUND OUT THAT THE columns 'oldbalanceOrg' and 'oldbalanceDest' have high vif factor and dropping them 
            enhances the vif factors of the columns. therefore will drop them from the dataframe"""
            
            numerical_columns.remove('oldbalanceOrg')
            numerical_columns.remove('oldbalanceDest')
            numerical_columns.remove(target_column_name)
            
            
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', RobustScaler())
            ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns)
            ])
            return preprocessing

        except Exception as e:
            raise BankingException(e,sys) from e   


    def data_transformation(self):
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()
            

            schema_file_path=self.schema_file_path 
            dataset_schema = read_yaml_file(file_path=schema_file_path)
            
            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            numerical_columns.remove('oldbalanceOrg')
            numerical_columns.remove('oldbalanceDest')
            target_column_name = dataset_schema[TARGET_COLUMN_KEY]
            
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]


            logging.info(f"Obtaining training and test file path.")


            schema_file_path = self.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            
            df = load_data(file_path=self.csv_file_path, schema_file_path=schema_file_path)
            df = df.drop(categorical_columns, axis = 1)
            df = df[numerical_columns]
            

            
            logging.info(f"Categorical columns dropped: {categorical_columns}")
            
            numerical_columns.remove(target_column_name)
            
            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_df = df.drop(columns=[target_column_name],axis=1)
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_df)

            transformed_df_directory= os.path.join('batch_prediction','array')
            os.makedirs(transformed_df_directory,exist_ok=True)

            train_file_name = os.path.basename(self.csv_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_df_directory, train_file_name)


            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=input_feature_train_arr)

            return transformed_train_file_path
        except Exception as e:
            raise BankingException(e,sys) from e



    def __del__(self):
        logging.info(f"{'='*20}Data Prediction log completed.{'='*20} \n\n")