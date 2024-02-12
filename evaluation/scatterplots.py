import numpy as np
import matplotlib.pyplot as plt
import os

output_folder = "./scatterplots"

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

mkdir(output_folder)

def create_scatterplots(real_data, syn_data, labels):
    num_features = real_data.shape[1]

    # Check if the shapes match
    if real_data.shape != syn_data.shape:
        raise ValueError("Input arrays must have the same shape")

    for feature_idx in range(num_features):
        plt.figure(figsize=(8, 6))
        plt.scatter(real_data[:, feature_idx], syn_data[:, feature_idx], s=20, alpha=0.6)
        plt.title(labels[feature_idx], fontsize=20)
        plt.xlabel("Ground Truth", fontsize=20)
        plt.ylabel("Predicted", fontsize=20)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, labels[feature_idx]+".png"))


real_data = np.load("C_real.py.npy")
syn_data = np.load("C_syn.py.npy")

real_data = np.concatenate((real_data, np.load("D_real.py.npy")), axis=0)
syn_data = np.concatenate((syn_data, np.load("D_syn.py.npy")), axis=0)
#
# real_data = np.concatenate((real_data, np.load("A_real.py.npy")), axis=0)
# syn_data = np.concatenate((syn_data, np.load("A_syn.py.npy")), axis=0)

real_data = np.concatenate((real_data, np.load("B_real.py.npy")), axis=0)
syn_data = np.concatenate((syn_data, np.load("B_syn.py.npy")), axis=0)

feature_labels = "SMAa,CD11b,CD44,CD31,CDK4,YKL40,CD11c,HIF1a,CD24,TMEM119,OLIG2,GFAP,VISTA,IBA1,CD206,PTEN,NESTIN,TCIRG1,CD74,MET,P2RY12,CD163,S100B,cMYC,pERK,EGFR,SOX2,HLADR,PDGFRa,MCT4,DNA1,DNA3,MHCI,CD68,CD14,KI67,CD16,SOX10"
feature_labels = feature_labels.split(",")

create_scatterplots(real_data, syn_data, feature_labels)
