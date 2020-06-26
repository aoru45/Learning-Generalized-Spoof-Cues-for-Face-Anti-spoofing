import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np
import cv2 as cv
import math
import sys
import glob
from config import *
from torchvision.utils import save_image
from unet import TripUNet
class MyModel(nn.Module):
    def __init__(self):
        self.net = ModifiedUNet(n_channels = 3, n_classes = 3)
        self.net.eval()
    def forward(self,anchor, positive, negative):
        regression_anchor, classification_anchor, _ = self.net(anchor) # (b,3,h,w) and (b,2)
        regression_positive, classification_positive, _ = self.net(positive)
        regression_negative, classification_negative, _ = self.net(negative)
        return [regression_anchor, regression_positive, regression_negative] , [classification_anchor, classification_positive, classification_negative]

def glob2(pattern1, pattern2):
    files = glob.glob(pattern1)
    files.extend(glob.glob(pattern2))
    return files
class Traindataset(Dataset):
    def __init__(self,root = "/ssd/xingduan/BCTC_ALL/Data",sub_dirs = ["2D_Plane","2D_Plane_Mask", "3D_Head_Model_Silicone", "3D_Head_Model_Wax", "Half_Mask"]):
        self.root = root
        self.sub_dirs = sub_dirs
        self.pos_filelist = {
            "liveness": glob2("{}/{}/*_rgb.jpg".format(root, "Live_Person"), "{}/{}/*_ir.jpg".format(root, "Live_Person"))
        }
        self.neg_filelist = {
            sub_dir: glob2("{}/{}/*_rgb.jpg".format(root, sub_dir), "{}/{}/*_ir.jpg".format(root, sub_dir)) for sub_dir in sub_dirs
        }
        self.transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])


    def __getitem__(self,idx):
        imgs = []
        for k in range(3):
            if k == 0:
                t = random.randint(0, len(self.pos_filelist["liveness"]) -1)
                l = self.pos_filelist["liveness"][t].split() # 取一个正样本
            elif k == 1:
                t = random.randint(0, len(self.pos_filelist["liveness"]) -1)
                l = self.pos_filelist["liveness"][t].split()
            else:
                key = random.choice(self.sub_dirs)
                t = random.randint(0, len(self.neg_filelist[key]) -1)
                l = self.neg_filelist[key][t].split() # 从所有类型的负样本中随机选取一个
            img_path = l[0]

            img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
            
            img_w, img_h = img.size
        
            ymin,ymax,xmin,xmax = 92, 188, 42, 138 # crop 整张脸

            img = img.crop([xmin,ymin,xmax,ymax])

            img = self.transform(img)

            imgs.append(img)
    
        return imgs[0], imgs[1], imgs[2]
    def __len__(self):
        return 20000


if __name__ == "__main__":
    train_dataset = Traindataset()
    model = MyModel()
    model.eval()
    model.load_state_dict(torch.load("./ckpt/149.pth"))
    for i in range(30):
        anchor, positive, negative = train_dataset[0]
        reg,cla = model(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))

        cla_anchor, cla_positive, cla_negative = cla
        save_image(torch.cat([reg], dim= 0),"a.jpg")

        print(cla_anchor.data, cla_positive.data, cla_negative.data)

        img_anchor = (anchor.permute((1,2,0)).data.numpy() * 255).astype(np.uint8)
        img_positive = (positive.permute((1,2,0)).data.numpy() * 255).astype(np.uint8)
        img_negative = (negative.permute((1,2,0)).data.numpy() * 255).astype(np.uint8)
        cv.imshow("img_anchor", cv.cvtColor(img_anchor, cv.COLOR_RGB2BGR))
        cv.imshow("img_positive", cv.cvtColor(img_positive, cv.COLOR_RGB2BGR))
        cv.imshow("img_negative", cv.cvtColor(img_negative, cv.COLOR_RGB2BGR))
        key = cv.waitKey(0)
        if key == ord("q"):
            break
    
    cv.destroyAllWindows()
