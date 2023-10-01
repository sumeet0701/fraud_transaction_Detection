import os
from pathlib import Path
import logging

while True:
    project_name = input("Enter the Project Name: \n")
    if project_name != " ":
        break


logging.info(f"creating project by name {project_name}")

list_files = [
    ".github/workflows/.gitkeep",
    ".github/workflows/main.yaml",
    "config/config.yaml",
    "config/schema.yaml",
    "config/model.yaml",
    "predictions/batch_prediction.py",
    "predictions/instance_prediction.py",
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/modal_pusher.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/Configuration.py",
    f"{project_name}/constant/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/utils.py",
    f"config/config.yaml",
    "schema.yaml",
    "requirements.txt",
    "setup.py",
    "main.py",
    "README.md"
]


for filepath in list_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok= True)
        logging.info(f"Creating a new directory at : {filedir} for file: {filename}")
    
    if (not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
        with open(filepath,"w") as f:
            pass
        logging.info(f"Creating a new file: {filename} for path: {filepath}")
    else:
        logging.info(f"file is already present at: {filepath}")