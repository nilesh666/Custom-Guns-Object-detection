import os
import torch
from torch.utils.data import DataLoader, random_split
from utils.logger import logging
from utils.custom_exception import CustomException
import sys
from torch import optim
from src.model_architecture import FasterRCNNModel
from src.data_processing import GunData

model_save_path = "artifacts/model"
os.makedirs(model_save_path, exist_ok=True)

class ModelTraining:
    def __init__(self, num_classes, epochs, model_class, dataset, learning_rate, device):
        self.num_classes = num_classes
        self.epochs = epochs
        self.model_class = model_class
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.device = device

        try:
            self.model = self.model_class(self.num_classes, self.device).model
            self.model.to(device)

            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        except Exception as e:
            logging.error("Error creating the model in model_training.py")
            raise CustomException(e, sys)
