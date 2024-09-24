import mat73
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from probreg import cpd
import copy
import pickle
from compute_registration_map import create_image

with open('registered_points.pkl', 'rb') as file:
    registered_points = pickle.load(file)
result_image = create_image(registered_points)
result_image.save('registered_spectral_map.png')
exit()