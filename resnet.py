import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
# from torchsummary import summary
import torchvision.transforms as T
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.metrics import r2_score
import csv
import configparser
import pandas as pd
import shutil

class ResNet(torch.nn.Module):

    def __init__(self, model_name=''):
        super(ResNet, self).__init__()

        self.feature_extractor = models.resnet50(pretrained=True) #try shufflenet, efficientnet

        columns_to_count = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10"
        columns_to_count = columns_to_count.split(",")
        num_output_features = len(columns_to_count)

        num_features = 64
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, num_features)

        # if(int(parameters['restore_featurextractor'])):
        #     self.feature_extractor.load_state_dict(torch.load(os.path.join(parameters['pretrained_featurextractor_path'], model_name + "_resnet.pt")))
        #     print("Features loaded from ",os.path.join(parameters['pretrained_featurextractor_path'], model_name + "_feature_extractor.pt"))
        #
        # if (int(parameters['freeze_featurextractor'])):
        #     for param in self.feature_extractor.parameters():
        #         param.requires_grad = False  # If want to freeze resnet weights

        self.final_layer = nn.Linear(num_features, num_output_features)
        # self.final_activation = nn.ReLU()

    def forward(self, images, edge_index=''):
        x = self.feature_extractor(images)
        x = self.final_layer(x)
        # x = self.final_activation(x)
        return x