import torch
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.custom_exception import CustomException
import sys
from utils.logger import logging
from tqdm import tqdm

class FasterRCNNModel:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.optimizer = None
        self.model = self.create_model().to(self.device)
        logging.info("Model intialized")
    
    def create_model(self):
        try:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            return model
        except Exception as e:
            logging.error("Error creating the model")
            raise CustomException(e, sys)

    def compile(self):
        try:
            self.optimizer = Adam(self.model.paramters(), lr = 1e-4)
            logging.info("Compiled the model")
        except Exception as e:
            logging.error("Error compiling the model")
            raise CustomException(e, sys)
        
    def train(self, train_loader, num_epochs):
        try:
            self.model.train()
            
            for i in range(1, num_epochs+1):
                train_loss = 0
                for img, tgt in tqdm(train_loader, desc=f"Epoch: {i}"):
                    img = [image.to(self.device) for image in img]
                    tgt = [{k:v.to(self.device) for k,v in t.iems()} for t in tgt]
                    loss_dict = self.model(img, tgt)
                    loss = sum(loss for loss in loss_dict.values())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss+=loss.items()
                logging.info(f"Epoch :{i} completed with train_loss: {total_loss}")

        except Exception as e:
            logging.error("Error training the model")
            raise CustomException(e, sys)