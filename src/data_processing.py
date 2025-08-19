import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils.logger import logging
from utils.custom_exception import CustomException
import sys

class GunData(Dataset):
    def __init__(self, root:str, device:str="cpu"):
        self.device = device
        self.image_path = os.path.join(root, "Images")
        self.labels_path = os.path.join(root, "Labels")
        self.images = sorted(os.listdir(self.image_path))
        self.labels = sorted(os.listdir(self.labels_path))

        logging.info("Data processing initialized")

    def __getitem__(self,idx):
        try:
            img = cv2.imread(os.path.join(self.image_path, str(self.images[idx])))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_res = img_rgb/255
            img_res = torch.as_tensor(img_res).permute(2,0,1)

            label_name = self.images[idx][:-4]+"txt"
            label = os.path.join(str(self.labels_path) ,str(label_name))

            if not os.path.exists(label):
                raise FileNotFoundError(f"No such file exists: {label}")
            
            target = {
                "boxes": torch.tensor([]),
                "area": torch.tensor([]),
                "image_id": torch.tensor([idx]),
                "labels": torch.tensor([], dtype=torch.int64)
            }

            with open(label, 'r') as l:
                c = int(l.readline())
                box=[list(map(int, l.readline().split()))]
                # for i in range(c):
                #     box.append(list(map(int, l.readline().split())))     

            if box:
                area = [(b[2]-b[0])*(b[3]-b[1]) for b in box]
                labels = [1]*len(box)
                target["boxes"] = torch.tensor(box, dtype=torch.float32)
                target["area"] = torch.tensor(area, dtype=torch.float32)
                target["labels"] = torch.tensor(labels, dtype=torch.int64)
                
            img_res = img_res.to(self.device)
            for key in target:
                target[key] = target[key].to(self.device)
            
            logging.info("Data processing completed")

            return img_res, target
        
        except Exception as e:
            logging.error("Error in data processing")
            raise CustomException(e, sys)
        

    def __len__(self):
        return len(self.images)
    
if __name__=="__main__":
    root = "artifacts/raw"    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = GunData(root, device)
    img, tgt = a[0]
    print("Img shape: ", img.shape)
    print("Tgt keys: ", tgt.keys())
    print("BBs : ", tgt["boxes"])