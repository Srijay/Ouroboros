import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths for the datasets
generated_file1 = r'F:\Datasets\RA\SpecTX\Srijay\results\generation\hyperion_generation_fromsinglevector_biggan_3_A_1\morph_features_generated_A.csv'
generated_file2 = r'F:\Datasets\RA\SpecTX\Srijay\results\generation\hyperion_generation_fromsinglevector_biggan_3_B_1\morph_features_generated_B.csv'
generated_file3 = r'F:\Datasets\RA\SpecTX\Srijay\results\generation\hyperion_generation_fromsinglevector_biggan_3_C_1\morph_features_generated_C.csv'
generated_file4 = r'F:\Datasets\RA\SpecTX\Srijay\results\generation\hyperion_generation_fromsinglevector_biggan_3_D_1\morph_features_generated_D.csv'

true_file = r'F:\Datasets\RA\SpecTX\Srijay\morph_features_all.csv'

# Columns in the dataset that are not features
non_feats = ['Unnamed: 0', 'spot_name', 'image_id', 'VisSpot']

# Loading the generated and true data into Pandas DataFrames
G1 = pd.read_csv(generated_file1)
G2 = pd.read_csv(generated_file2)
G3 = pd.read_csv(generated_file3)
G4 = pd.read_csv(generated_file4)

G = pd.concat([G1,G2,G3,G4])

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

# Applying Standard Scaler to normalize the data
ss = StandardScaler().fit(X)

# Transforming both datasets using the fitted scaler
Gft, Tft = ss.transform(Gf), ss.transform(Tf)

# Extracting the number of features
n_feats = len(feats)

# Converting the transformed data back to Pandas DataFrames
Gft = pd.DataFrame(Gft[:, :n_feats], index=Gf.index)
Tft = pd.DataFrame(Tft[:, :n_feats], index=Tf.index)

# Merging the two datasets on their index
df = pd.merge(Tft, Gft, left_index=True, right_index=True)

X = np.array(df)

# Splitting the merged data back into 'true' and 'generated' datasets
Xt, Xg = X[:, :n_feats], X[:, n_feats:]

# Calculating the Euclidean distance between corresponding rows in Xt and Xg
D = np.linalg.norm(Xt - Xg, axis=1)

#%%
def average_distance(matrix, M):
    """
    Function to calculate the average distance of each row in a matrix to M randomly selected other rows.
    :param matrix: NumPy array representing the data matrix.
    :param M: Number of random rows to select for distance calculation.
    :return: Array of average distances.
    """
    N, d = matrix.shape
    avg_distances = np.zeros(N)

    for i in range(N):
        # Select M random rows, excluding the current row i
        random_indices = np.random.choice(np.delete(np.arange(N), i), M, replace=False)
        selected_rows = matrix[random_indices]

        # Calculate distances from row i to the selected M rows
        distances = np.linalg.norm(matrix[i] - selected_rows, axis=1)

        # Compute the average distance
        avg_distances[i] = np.mean(distances)

    return avg_distances

# Calculating the average distances for the 'true' dataset
Dbar = average_distance(Xt, M=100)

#%%
# Creating a boxplot to compare distributions of D and Dbar
plt.boxplot([D, Dbar], showfliers=False)
# plt.title("Morphological Feature Distance Comparison")
plt.ylabel("Morphological Feature Distance", fontsize=12)
plt.xticks([1,2],["Ouroboros", "Random"], fontsize=12)
plt.yticks(fontsize=12)

from scipy import stats

# Performing a Wilcoxon signed-rank test on D and Dbar.
# Hypothesis: For each image patch, the distance between morphological features
# of cells in the original image and its generated counterpart (D) is less than
# the average distance between morphological features of cells in that image
# and randomly selected original patches from the same whole slide image (Dbar).
# This test aims to statistically verify if the generated image patches
# maintain closer morphological feature similarity to their original counterparts
# compared to random original patches, indicating accuracy in feature preservation during generation.
w_stat, p_value = stats.wilcoxon(D, Dbar, alternative='less')

# Printing the test statistics
print("W-statistic:", w_stat)
print("P-value:", p_value)
plt.show()