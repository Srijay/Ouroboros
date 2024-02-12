import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths for the datasets
generated_file1 = r'C:\Users\Srijay\Desktop\Projects\SpecTx-Generative\interpolation\morph_features.csv'
true_file = r'F:\Datasets\RA\SpecTX\Srijay\morph_features_all.csv'

reference_spot_id = '1_real'

# Columns in the dataset that are not features
non_feats = ['Unnamed: 0', 'spot_name', 'image_id', 'VisSpot']

# Loading the generated and true data into Pandas DataFrames
G1 = pd.read_csv(generated_file1)
G = G1
T = pd.read_csv(true_file)


# Extracting the feature columns
feats = G.keys().difference(non_feats)

feats = [
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Angular second moment (F0)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Contrast (F1)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Correlation (F2)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Difference entropy (F10)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Difference variance (F9)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Entropy (F8)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Information measure of correlation 1 (F11)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Information measure of correlation 2 (F12)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Inverse difference moment (F4)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Sum average (F5)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Sum entropy (F7)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Sum of squares (F3)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Haralick Sum variance (F6)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Mean',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Eosin: Std.dev.',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Angular second moment (F0)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Contrast (F1)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Correlation (F2)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Difference entropy (F10)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Difference variance (F9)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Entropy (F8)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Information measure of correlation 1 (F11)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Information measure of correlation 2 (F12)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Inverse difference moment (F4)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Sum average (F5)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Sum entropy (F7)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Sum of squares (F3)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Haralick Sum variance (F6)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Mean',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: Hematoxylin: Std.dev.',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Angular second moment (F0)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Contrast (F1)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Correlation (F2)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Difference entropy (F10)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Difference variance (F9)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Entropy (F8)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Information measure of correlation 1 (F11)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Information measure of correlation 2 (F12)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Inverse difference moment (F4)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Sum average (F5)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Sum entropy (F7)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Sum of squares (F3)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Haralick Sum variance (F6)',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Mean',
       'Circle: Diameter 50.0 Âµm: 0.44 Âµm per pixel: OD Sum: Std.dev.',
       'Circularity',
       'Detection probability',
       'Eosin: Max', 'Eosin: Mean',
       'Eosin: Median', 'Eosin: Min', 'Eosin: Std.Dev.', 'Hematoxylin: Max',
       'Hematoxylin: Mean', 'Hematoxylin: Median', 'Hematoxylin: Min',
       'Hematoxylin: Std.Dev.', 'Length Âµm',
       'Max diameter Âµm',
       'Min diameter Âµm',
       'ROI: 0.44 Âµm per pixel: OD Sum: Mean',
       'Solidity', 'Area Âµm^2'
        ]


# Grouping data by 'spot_name' and calculating the mean of features
Gf = G.groupby('spot_name')[feats].mean()
Tf = T.groupby('spot_name')[feats].mean()

#%%
from sklearn.preprocessing import StandardScaler

# Stacking the feature means vertically to form a single dataset for scaling
X = np.vstack((Gf, Tf))
ss = StandardScaler().fit(X)
Gft, Tft = ss.transform(Gf), ss.transform(Tf)

# Extracting the number of features
n_feats = len(feats)

# Converting the transformed data back to Pandas DataFrames
Gft = pd.DataFrame(Gft[:, :n_feats], index=Gf.index)

row_to_compare = Gft.loc[reference_spot_id].values.reshape(1, -1)
distances = np.linalg.norm(Gft.values - row_to_compare, axis=1)

result_df = pd.DataFrame(distances, index=Gft.index, columns=['Distance'])
print(result_df)