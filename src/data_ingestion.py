import os
import sys
import kagglehub
import zipfile
import shutil
from utils.logger import logging
from utils.custom_exception import CustomException
from config.data_ingestion_config import *

class DataIngestion:
    def __init__(self, dataset_name:str, target_path:str):
        self.dataset_name = dataset_name
        self.target_path = target_path

    def create_raw(self):
        try:
            raw_dir = os.path.join(TARGET_DIR, "raw")
            if not os.path.exists(raw_dir):
                os.makedirs(raw_dir)
            return raw_dir
        except Exception as e:
            logging.error("Error in creating raw dir")
            raise CustomException(e, sys)

    def extract_data(self, path:str, raw_dir:str):
        try:
            if path.endswith(".zip"):
                with zipfile.ZipFile(path, 'r') as z:
                    z.extractall(path)
            
            images_folder = os.path.join(path, "Images")
            labels_folder = os.path.join(path, "Labels")

            if os.path.exists(images_folder):
                shutil.move(images_folder, raw_dir)
                logging.info("Images stored in raw dir successfully")
            else:
                logging.info("Images folder does not exist")

            if os.path.exists(labels_folder):
                shutil.move(labels_folder, raw_dir)
                logging.info("Labels stored in raw dir successfully")
            else:
                logging.info("Labels folder does not exist")

        except Exception as e:
            logging.error("Error in extracting from zip file")
            raise CustomException(e, sys)
    
    def download(self, raw_dir:str):
        try:

            path = kagglehub.dataset_download(self.dataset_name)
            logging.info("Downloading data from Kaggle")
            self.extract_data(path , raw_dir)

        except Exception as e:
            logging.error("Error in downloading from kaggle")
            raise CustomException(e, sys)

    def run(self):
        try:
            raw_dir = self.create_raw()
            logging.info("Created raw dir")

            self.download(raw_dir)
            logging.info("Donwloaded data from kaggle")

        except Exception as e:
            logging.error("Error in downloading and extracting the data")
            raise CustomException(e, sys)


if __name__=="__main__":
    a = DataIngestion(DATASET_NAME, TARGET_DIR)
    a.run()

