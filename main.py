from flask import Flask, request
import sys
import pandas as pd
import pip
from fraud.utils.utils import read_yaml_file, write_yaml_file,load_numpy_array_data,load_object
from matplotlib.style import context
from fraud.logger import logging
from fraud.exception import FraudException
import os, sys
import json
from werkzeug.utils import secure_filename
from fraud.entity.artifact_entity import ModelPusherArtifact
from fraud.config.Configuration import Configuartion
from fraud.constant import CONFIG_DIR, get_current_time_stamp
from fraud.pipeline.pipeline import Pipeline
from fraud.entity.banking_predictor import BankingPredictor, BankingData
from flask import send_file, abort, render_template
from predictions.batch_prediction import Batch_prediction

pipeline = Pipeline()
ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "banking"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from fraud.logger import get_log_dataframe

BANKING_DATA_KEY = "banking_data"
IS_FRAUD_VALUE_KEY = "fraud"

app = Flask(__name__)



UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/train', methods=['GET', 'POST'])
def train():
    
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    
    pipeline.run_pipeline()
    message='Training Completed'

    return render_template('index.html', message=message)


@app.route('/instance', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        context = {
            BANKING_DATA_KEY: None,
            IS_FRAUD_VALUE_KEY: None
        }
        step = float(request.form['step'])
        amount = float(request.form['amount'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        newbalanceDest = float(request.form['newbalanceDest'])
        isFlaggedFraud = float(request.form['isFlaggedFraud'])
    

        banking_data = BankingData(step=step,
                                    amount=amount,
                                    newbalanceOrig=newbalanceOrig,
                                    newbalanceDest=newbalanceDest,
                                    isFlaggedFraud=isFlaggedFraud
                                    )
        banking_df = banking_data.get_banking_input_data_frame()
        fraud_predictor = BankingPredictor(model_dir=MODEL_DIR)

        
        is_Fraud_value = fraud_predictor.predict(X=banking_df)
        context = {
            BANKING_DATA_KEY: banking_data.get_banking_data_as_dict(),
            IS_FRAUD_VALUE_KEY: is_Fraud_value
        }



        
    return render_template("predict.html",is_Fraud_value=context[IS_FRAUD_VALUE_KEY])

ALLOWED_EXTENSIONS={'csv'}
@app.route('/batch', methods=['GET', 'POST'])
def perform_batch_prediction():
    if request.method == 'GET':
        return render_template('batch.html')
    else:
    
        file = request.files['csv_file']  # Update the key to 'csv_file'
            # Directory path
        directory_path = UPLOAD_FOLDER
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

        
        # Check if the file has a valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the new file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)
            
            logging.info("CSV received and Uploaded")

            # Perform batch prediction using the uploaded file
            batch = Batch_prediction(file_path)
            array_file_path=batch.data_transformation()
            
            array=load_numpy_array_data(array_file_path)
            
            
            logging.info(f"Loaded numpy from batch prediciton :{array}")
            
            logging.info(f"Array Shape :{array.shape}")
            
            columns=['step','amount','newbalanceOrig','newbalanceDest','isFlaggedFraud']
            
            df = pd.DataFrame(array, columns=columns)
            
            predictor=BankingPredictor(model_dir=MODEL_DIR)
            model_path=predictor.get_latest_model_path()
            model=load_object(model_path)
            
            logging.info(f" Loading model from path : {model_path}")
            
            predictions=model.predict(df)
            
            prediction_df = pd.DataFrame(predictions, columns=['prediction'])
            
            file_path=os.path.join('batch_prediction','prediction')
            os.makedirs(file_path)
            prediction_csv_file_path=os.path.join(file_path,'predictions.csv')
            
            prediction_df.to_csv(prediction_csv_file_path)
            

            output = "Batch Prediction Done"
            return render_template("batch.html", prediction_result=output, prediction_type='batch')
        else:
            return render_template('batch.html', prediction_type='batch', error='Invalid file type')
        
        
        

if __name__ == "__main__":
    app.run(port=8000)