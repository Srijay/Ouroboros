import numpy as np
from pathlib import Path
import gzip
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import pickle
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

images_dir = "./data/images"
image_names = ['A','B','C','D']
extracted_patches_dir = r"./data/patches"
protein_expressions_csv = "./data/protein_expressions_spotlevel.csv"
protein_expressions_df = pd.read_csv(protein_expressions_csv,sep=',')

mkdirs(extracted_patches_dir)

win = 256
hw = win//2

for imagename in image_names:
    imagefilename = os.path.join(images_dir, imagename+'.png')
    print("Processing image file ",imagefilename)
    protein_expressions_df_current = protein_expressions_df[protein_expressions_df['img']==imagename+'1']
    I = imread(imagefilename)
    for idx, row in protein_expressions_df_current.iterrows():
        patch_name = imagename + "_" + str(row['array_row']) + 'x' + str(row['array_col'])+".png"
        y, x = int(row['hne_row_loc']), int(row['hne_col_loc'])
        if(y == 0 and x == 0):
            continue
        if(int(row['in_tissue'])==1):
            if (y-hw)<0 or (y+hw)>I.shape[0] or (x-hw)<0 or (x+hw)>I.shape[1]: continue
            p = I[y-hw:y+hw,x-hw:x+hw]
            imsave(os.path.join(extracted_patches_dir,patch_name), p)