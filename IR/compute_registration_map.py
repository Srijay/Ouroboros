import mat73
from PIL import Image, ImageDraw
from probreg import cpd
import copy

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
# hne_df = hne_df[hne_df['img']=='A1']
# size=(17966, 22915)
# coordinates_array = hne_df[['hne_col_loc', 'hne_row_loc']].to_numpy()
# hne_coordinates_list = coordinates_array.tolist()
# result_image = create_image(hne_coordinates_list, radius=64, size=size)
# result_image.save('hne_map_highres.png')
# hne_coordinates_list = [[int(element / 13) for element in sublist] for sublist in hne_coordinates_list]
# result_image = create_image(hne_coordinates_list, size=(int(size[0]/13), int(size[1]/13)))
# result_image.save('hne_map_lowres.png')
# # hne_coordinates_list = [[145, 300], [151, 368], [150, 452], [194, 778], [260, 1124], [428, 1180], [441, 1183], [448, 1200], [478, 1211], [495, 1213], [447, 1194], [473, 1179], [516, 1151], [569, 1169], [587, 1149], [639, 1168], [1254, 1073], [1260, 979], [1306, 718], [1318, 589], [1293, 424], [1236, 290], [1232, 176], [1150, 179], [1103, 212], [1101, 172], [1048, 152], [788, 204], [841, 245], [895, 286], [800, 248], [718, 197], [358, 258], [155, 286]]

# Mapping for C
hne_coordinates_list = [[304, 995], [491, 1238], [1047, 1066], [449, 194], [513, 87], [1117, 413], [1086, 205]]
spectral_coordinates_list = [[81, 951], [265, 1219], [900, 1136], [404, 141], [380, 54], [1095, 499], [1090, 233]]

# Mapping for A
hne_coordinates_list = []
spectral_coordinates_list = []

# result_image = create_image(hne_coordinates_list, size=(int(size[0]/13), int(size[1]/13)))
# result_image.save('hne_map_lowres_points.png')

#hyperion_mapping = {'A1': 'N540', 'B1': 'N935', 'C1': 'N403', 'D1': 'N1027'}

data_dict = mat73.loadmat(r'F:\Datasets\RA\SpecTX\Krupakar\IR\n403_18e2_EMSC.mat')
l = data_dict['n403_18e2_EMSC']['xy']

# data_dict = scipy.io.loadmat(r'F:\Datasets\RA\SpecTX\Krupakar\IR\n935_19b1_EMSC.mat')
spectral_coordinates_list_real = data_dict['n540_19a2_EMSC']['xy'].tolist()

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
# result_image = create_image(spectral_coordinates_list)
# result_image.save('spectral_map_points.png')

print('registration process started')
tf_param, _, _ = cpd.registration_cpd(spectral_coordinates_list, hne_coordinates_list)
transposed_coordinates_ = copy.deepcopy(transposed_coordinates)
result = tf_param.transform(transposed_coordinates_)

registration_dict = {tuple(spectral_coordinates_list_real[i]): tuple(result[i]) for i in range(len(spectral_coordinates_list_real))}

# with open('registered_points_a.pkl', 'wb') as file:
#     pickle.dump(result, file)
#
# with open('registered_dict_a.pkl', 'wb') as file:
#     pickle.dump(registration_dict, file)

result_image = create_image(result)
result_image.save('registered_spectral_map.png')