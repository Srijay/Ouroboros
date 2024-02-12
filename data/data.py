import glob
import os
import pandas as pd
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from scipy.spatial import distance
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import tifffile
import configparser
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from combat.pycombat import pycombat

def ZSSTransform(z):
    zlog = np.log10(z[z>0])
    tzlog = (zlog-np.mean(zlog))/np.std(zlog)
    tzlog_min = np.min(tzlog)-1
    tz = z.copy()
    tz[tz>0] = tzlog
    tz[tz==0] = tzlog_min
    return tz

class HyperionDataset(Dataset):

    def __init__(self, image_dir, mode, perturb_biomarker="CD31"):

        super(Dataset, self).__init__()

         # Z transformation and batch correction
        cell_counts_path = r"protein_expressions_celllevel.csv"
        cell_df = pd.read_csv(cell_counts_path)
        cell_df = cell_df[cell_df['VisSpot'].notna()]
        wsi_image_ids = cell_df.VisSpot.apply(lambda x: str(x)[-2:])  # get the image IDs
        cell_df = cell_df[['VisSpot','Location_Center_Y','Location_Center_X'] + self.biomarkers]
        cell_df[biomarkers] = cell_df[biomarkers].apply(ZSSTransform)  # apply the log transform
        cell_df[biomarkers] = pycombat(cell_df[biomarkers].T, wsi_image_ids).T  # batch correction
        cell_df = cell_df.groupby('VisSpot', as_index=False).mean()

        self.image_dir = image_dir
        self.mode = mode

        spot_ids = cell_df['VisSpot'].to_list()
        test_ids = [spot_id for spot_id in spot_ids if spot_id.split("-")[2] in ['A1']]
        train_ids = [item for item in spot_ids if item not in test_ids]

        self.spot_ids = train_ids if self.mode=='train' else test_ids

        biomarkers = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10"
        biomarkers = biomarkers.split(",")
    
        self.hyperion_features = cell_df
        self.biomarkers = biomarkers
        
        self.perturb_biomarker = perturb_biomarker


    def read_image(self,img_path):
        img = Image.open(img_path)
        return np.asarray(img)

    def __len__(self):
        return len(self.spot_ids)

    def __getitem__(self, index):

        spot_id = self.spot_ids[index]

        image_id = spot_id.split("-")[2]

        loc_id = self.hyperion_features.loc[self.hyperion_features['VisSpot'] == spot_id, 'id'].values[0]

        img_name = image_id+"_"+loc_id+".png"

        image_path = os.path.join(self.image_dir,img_name)
        image = self.read_image(image_path)
        image = image/255.0
        image = image[:, :, :3]

        transform = T.Compose([T.ToTensor()])
        image = transform(image)

        hyperion_features = self.hyperion_features[self.hyperion_features['VisSpot'] == spot_id]
        hyperion_features = hyperion_features[self.biomarkers].values.flatten().tolist()

        hyperion_features = np.array(hyperion_features)

        return img_name, hyperion_features, image
