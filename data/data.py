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

        self.image_dir = image_dir
        self.mode = mode
        hyp_counts_path = r"F:\Datasets\RA\SpecTX\Srijay\protein_expressions_spotlevel.csv"

        self.df_hyperion = pd.read_csv(hyp_counts_path)

        spot_ids = self.df_hyperion['VisSpot'].to_list()
        test_ids = [spot_id for spot_id in spot_ids if spot_id.split("-")[2] in ['A1']]
        train_ids = [item for item in spot_ids if item not in test_ids]

        self.spot_ids = train_ids if self.mode=='train' else test_ids

        config = configparser.ConfigParser()
        config.read(r'C:\Users\Srijay\Desktop\Projects\Transcriptomics\cell_types')
        parameters = config['celltypes']
        self.biomarkers = parameters['biomarkers'].split(",")

        # Z transformation and batch correction
        cell_counts_path = r"F:\Datasets\RA\SpecTX\Srijay\protein_expressions_celllevel.csv"
        cell_df = pd.read_csv(cell_counts_path)
        cell_df = cell_df[cell_df['VisSpot'].notna()]
        wsi_image_ids = cell_df.VisSpot.apply(lambda x: str(x)[-2:])  # get the image IDs
        biomarkers = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10"
        biomarkers = biomarkers.split(",")
        self.biomarkers = biomarkers
        cell_df = cell_df[['VisSpot','Location_Center_Y','Location_Center_X'] + biomarkers]
        cell_df[biomarkers] = cell_df[biomarkers].apply(ZSSTransform)  # apply the log transform
        cell_df[biomarkers] = pycombat(cell_df[biomarkers].T, wsi_image_ids).T  # batch correction
        # cell_df['image_id'] = cell_df['VisSpot'].str.split("-").str[2]
        # df2 = pd.read_csv(r'F:\Datasets\RA\SpecTX\Srijay\morph_features_all.csv')
        # df_new = pd.merge(cell_df, df2[['spot_name', 'VisSpot']], on='VisSpot', how='left')
        # output_path = r"F:\Datasets\RA\SpecTX\Srijay\cellular_proteinexpressions_batchcorrected.csv"
        # df_new.to_csv(output_path)
        # exit()

        cell_df = cell_df.groupby('VisSpot', as_index=False).mean()
        # output_path = r"F:\Datasets\RA\SpecTX\Srijay\hyperion_counts_biomarkers_batchcorrected.csv"
        # cell_df.to_csv(output_path)
        # exit()

        self.hyperion_features = cell_df

        # for operation in self.biomarkers:
        #     self.hyperion_features[operation] = (self.hyperion_features[operation] - self.hyperion_features[operation].min()) / (self.hyperion_features[operation].max() - self.hyperion_features[operation].min())

        self.perturb_biomarker = perturb_biomarker


    def read_image(self,img_path):
        img = Image.open(img_path)
        return np.asarray(img)

    def __len__(self):
        return len(self.spot_ids)

    def __getitem__(self, index):

        spot_id = self.spot_ids[index]
        # spot_id = "CCCTCGGGAGCCTTGT-1-A1"
        # spot_id = "GTCAGTTGTGCTCGTT-1-A1"

        image_id = spot_id.split("-")[2]

        loc_id = self.df_hyperion[(self.df_hyperion['VisSpot'] == spot_id)].id.values[0]

        img_name = image_id+"_"+loc_id+".png"

        image_path = os.path.join(self.image_dir,img_name)
        image = self.read_image(image_path)
        image = image/255.0
        image = image[:, :, :3]

        transform = T.Compose([T.ToTensor()])
        image = transform(image)

        hyperion_features = self.hyperion_features[self.hyperion_features['VisSpot'] == spot_id]
        hyperion_features = hyperion_features[self.biomarkers].values.flatten().tolist()
        # print(hyperion_features)
        # #Plot histogram
        # colors = plt.cm.gist_rainbow(np.linspace(0, 1, 38))
        # n, bins, patches = plt.hist(hyperion_features, bins=38)
        # for l, p in zip(self.biomarkers, patches):
        #     plt.text(p.get_x() + p.get_width() / 2, -1, l, fontsize='xx-small', rotation=90)
        # for c, p in zip(colors, patches):
        #     p.set_facecolor(c)


        hyperion_features = np.array(hyperion_features)

        # pooled_features = []
        # y_locations = []
        # x_locations = []
        # for i, row in hyperion_locations.iterrows():
        #     y_loc = int(row["Location_Center_Y"])
        #     x_loc = int(row["Location_Center_X"])
        #     scores = np.array(row[self.biomarkers].values.flatten().tolist())
        #     pooled_features.append(scores)
        #     y_locations.append(y_loc)
        #     x_locations.append(x_loc)
        #
        # try:
        #     y_locations = [int((x - min(y_locations))/ (max(y_locations) - min(y_locations))*255) for x in y_locations]
        # except:
        #     y_locations = [int((x/max(y_locations))*255) for x in y_locations]
        #
        # try:
        #     x_locations = [int((x - min(x_locations))/ (max(x_locations) - min(x_locations))*255) for x in x_locations]
        # except:
        #     x_locations = [int((x/max(x_locations))*255) for x in x_locations]
        #
        # locations = [list(a) for a in zip(y_locations, x_locations)]

        # image = Image.new('RGB', (256,256), color='white')
        # draw = ImageDraw.Draw(image)
        # dot_radius = 2
        # dot_color = 'black'
        # for location in locations:
        #     draw.ellipse([location[0] - dot_radius, location[1] - dot_radius,
        #                   location[0] + dot_radius, location[1] + dot_radius], fill=dot_color)
        # image.save('cellular_layout.png')
        # image.show()
        # exit()

        # perturb_image_index = 4 #Imp indices: 3,8,10
        # perturb_biomarkers = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10".split(",")

        # del pooled_features[perturb_image_index]
        # del locations[perturb_image_index]

        # perturb_biomarkers = self.perturb_biomarker.split(",")
        # for perturb_biomarker in perturb_biomarkers:
        #     pooled_features[perturb_image_index][self.biomarkers.index(perturb_biomarker)]=0
            # pooled_features[perturb_image_index][self.biomarkers.index(perturb_biomarker)]= pooled_features[10][self.biomarkers.index(perturb_biomarker)]

        # pooled_features = np.array(pooled_features)

        # positions = np.arange(1, 39)
        # biomarkers = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10".split(",")
        # plt.boxplot(plotfeatures, positions=positions, vert=True, patch_artist=True)
        # plt.xlabel('Protein Biomarkers')
        # plt.ylabel('Expresssion Values')
        # plt.xticks(list(range(1,39)), biomarkers, rotation=90)
        # plt.show()
        # exit()

        # locations = np.array(locations)

        return img_name, hyperion_features, image