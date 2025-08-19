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
            # logging.info(f"Model parameterts are : {self.model.parameters()}")
        except Exception as e:
            logging.error("Error creating the model in model_training.py")
            raise CustomException(e, sys)

    def collate_fn(self, batch):
        return zip(*(batch))
    
    def split_data(self):
        try:
            dataset = GunData(self.dataset, device=self.device)
            dataset = torch.utils.data.Subset(dataset, range(5))
            train_size = int(0.8*len(dataset))
            val_size = len(dataset) - train_size
            train_data, val_data = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_data, batch_size=3, shuffle=True,num_workers=0, collate_fn = self.collate_fn)
            val_loader = DataLoader(val_data, batch_size=3, shuffle=False,num_workers=0, collate_fn = self.collate_fn)

            logging.info("Splitted the data")
            return train_loader, val_loader

        except Exception as e:
            logging.error("Error splitting the data in model_training.py")
            raise CustomException(e, sys)
        
    def train(self):
        try:
            # total_loss=0
            train_loader, val_loader = self.split_data()
            for epoch in range(self.epochs):
                self.model.train()
                for i, (img,tgt) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    losses = self.model(img, tgt)

                    if isinstance(losses, dict):
                        total_loss=0
                        for k,v in losses.items():
                            if isinstance(v, torch.Tensor):
                                total_loss+=v


                        if total_loss==0:
                            logging.info("Error in loss calcultaion in model_training.py")
                            raise ValueError("Error in capturing loss in model_training.py")
                    else:
                        total_loss = losses[0]

                    total_loss.backward()
                    self.optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    for img, tgt in val_loader:
                        val_losses = self.model(img, tgt)
                        logging.info(f"Val loss: {val_losses}")
                
                model_path = os.path.join(self.dataset, "F_RCNN.pth")
                torch.save(self.model.state_dict(), model_path)
                logging.info("Model saved successfully")


        except Exception as e:
            logging.error("Error in training the model_training.py")
            raise CustomException(e, sys)
        
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = ModelTraining(
        num_classes=2, 
        epochs=1,
        model_class=FasterRCNNModel,
        dataset="artifacts/raw/",
        learning_rate=1e-4,
        device=device
    )

    a.train()
        