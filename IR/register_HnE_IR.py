import mat73
from PIL import Image, ImageDraw
from probreg import cpd
import copy
import pandas as pd

def create_image(coordinates, radius = 5, size=None):
    max_x = int(max(coordinates, key=lambda x: x[0])[0])
    max_y = int(max(coordinates, key=lambda x: x[1])[1])
    min_x = int(min(coordinates, key=lambda x: x[0])[0])
    min_y = int(min(coordinates, key=lambda x: x[1])[1])

    if(size):
        min_x = min_y = 0
    else:
        size = (max_x + radius, max_y + radius)
    image = Image.new('RGB', size, color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    for x, y in coordinates:
        bounding_box = (x, y, x + radius, y + radius)
        draw.ellipse(bounding_box, fill=(255, 255, 255))
    return image

# hne_df = r"F:\Datasets\RA\SpecTX\Srijay\hne_counts.csv"
# hne_df = pd.read_csv(hne_df,sep=',')
# hne_df = hne_df[hne_df['img']=='C1']
# size=(15921, 15768)
# coordinates_array = hne_df[['hne_col_loc', 'hne_row_loc']].to_numpy()
# hne_coordinates_list = coordinates_array.tolist()
# result_image = create_image(hne_coordinates_list, radius=64, size=size)
# result_image.save('hne_map_highres.png')
# hne_coordinates_list = [[int(element / 13) for element in sublist] for sublist in hne_coordinates_list]
# result_image = create_image(hne_coordinates_list, size=(int(size[0]/13), int(size[1]/13)))
# result_image.save('hne_map_lowres.png')

# Mapping for C
hne_coordinates_list = [[304, 995], [491, 1238], [1047, 1066], [449, 194], [513, 87], [1117, 413], [1086, 205]]
spectral_coordinates_list = [[81, 951], [265, 1219], [900, 1136], [404, 141], [380, 54], [1095, 499], [1090, 233]]

data_dict = mat73.loadmat(r'F:\Datasets\RA\SpecTX\Krupakar\IR\n403_18e2_EMSC.mat')
spectral_coordinates_list_real = data_dict['n403_18e2_EMSC']['xy'].tolist()

radius = 5
max_x, min_x = max(point[0] for point in spectral_coordinates_list_real), min(point[0] for point in spectral_coordinates_list_real)
max_y, min_y = max(point[1] for point in spectral_coordinates_list_real), min(point[1] for point in spectral_coordinates_list_real)
width = max_x + radius - min_x
height = max_y + radius - min_y

transposed_coordinates = []
for x, y in spectral_coordinates_list_real:
    transposed_x = y
    transposed_y = width - x
    transposed_coordinates.append([transposed_x, transposed_y])

result_image = create_image(spectral_coordinates_list_real, radius=1)
result_image.save('spectral_map.png')

print('registration process started')
tf_param, _, _ = cpd.registration_cpd(spectral_coordinates_list, hne_coordinates_list)
transposed_coordinates_ = copy.deepcopy(transposed_coordinates)
result = tf_param.transform(transposed_coordinates_)

registration_dict = {tuple(spectral_coordinates_list_real[i]): tuple(result[i]) for i in range(len(spectral_coordinates_list_real))}

result_image = create_image(result)
result_image.save('registered_spectral_map.png')