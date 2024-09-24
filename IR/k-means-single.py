import mat73
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import pickle

data_dict = mat73.loadmat(r'F:\Datasets\RA\SpecTX\Krupakar\IR\n403_18e2_EMSC.mat')
spectral_coordinates_list = data_dict['n403_18e2_EMSC']['xy'].tolist()[1:]
spectral_coordinates_list = np.array(spectral_coordinates_list)
spectra = np.array(data_dict['n403_18e2_EMSC']['Data'])
spectra = spectra.T

# Perform k-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(spectra)
labels = kmeans.labels_

# Define colors for clusters
cluster_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]

def create_image(coordinates, radius = 5, use_registered=False):
    max_x = int(max(coordinates, key=lambda x: x[0])[0])
    max_y = int(max(coordinates, key=lambda x: x[1])[1])
    min_x = int(min(coordinates, key=lambda x: x[0])[0])
    min_y = int(min(coordinates, key=lambda x: x[1])[1])
    size = (max_x + radius - min_x, max_y + radius - min_y)
    image = Image.new('RGB', size, color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    for i, (x, y) in enumerate(coordinates):
        bounding_box = (x - min_x, y - min_y, x - min_x + radius, y - min_y + radius)
        draw.ellipse(bounding_box, fill=cluster_colors[labels[i]])
    return image

transposed_coordinates = spectral_coordinates_list
# radius = 5
# max_x, min_x = max(point[0] for point in spectral_coordinates_list), min(point[0] for point in spectral_coordinates_list)
# max_y, min_y = max(point[1] for point in spectral_coordinates_list), min(point[1] for point in spectral_coordinates_list)
# width = max_x + radius - min_x
# height = max_y + radius - min_y
# transposed_coordinates = []
# for x, y in spectral_coordinates_list:
#     transposed_x = y
#     transposed_y = width - x
#     transposed_coordinates.append([transposed_x, transposed_y])

result_image = create_image(transposed_coordinates)
result_image.save('k_means_c.png')