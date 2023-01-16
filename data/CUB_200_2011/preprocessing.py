import os
import numpy as np
import pickle
from PIL import Image

data_path = './'
train_test_split_file = os.path.join(data_path, 'train_test_split.txt')
image_names_file = os.path.join(data_path, 'images.txt')
class_names_file = os.path.join(data_path, 'classes.txt')
labels_file = os.path.join(data_path, 'image_class_labels.txt')
bds_file = os.path.join(data_path, 'bounding_boxes.txt')
images_path = os.path.join(data_path, 'images')
save_train_test_id_path = os.path.join(data_path, 'train_test_ids.pkl')
save_image_path = os.path.join(data_path, 'data.pkl')
save_gt_path = os.path.join(data_path, 'gts.pkl')

# Dictionary : phase => image_ids
image_id = dict()
image_id['train'] = []
image_id['test'] = []

indices = np.loadtxt(train_test_split_file)
for index in indices:
    if index[1]:
        train_image_id = int(index[0])
        image_id['train'].append(train_image_id)
    else:
        test_image_id = int(index[0])
        image_id['test'].append(test_image_id)

with open(save_train_test_id_path, 'wb') as fw:
    pickle.dump(image_id, fw, pickle.HIGHEST_PROTOCOL)

# # Dictionary : class_id => class_name
# with open(class_names_file, 'r') as f:
#     class_infos = f.readlines()
#     class_pairs = [pair[:-1].split(' ') for pair in class_infos]

# class_names = dict()
# for p in class_pairs:
#     class_names[int(p[0])] = p[1]


# # Dictionary : image_id => image_file_name
# with open(image_names_file, 'r') as f:
#     image_infos = f.readlines()
#     image_pairs = [pair[:-1].split(' ') for pair in image_infos]

# image_names = dict()
# for p in image_pairs:
#     image_names[int(p[0])] = p[1]


# Dictionary : image_id => image_rgb
with open(image_names_file, 'r') as f:
    image_infos = f.readlines()
    image_pairs = [pair[:-1].split(' ') for pair in image_infos]

image_data = dict()
for p in image_pairs:
    image_data[int(p[0])] = Image.open(os.path.join(images_path, p[1]))

with open(save_image_path, 'wb') as fw:
    pickle.dump(image_data, fw, pickle.HIGHEST_PROTOCOL)

# Dictionary : image_id => [label, bounding_box]
gt = dict()

with open(labels_file, 'r') as f:
    labels_infos = f.readlines()
    labels_pairs = [pair[:-1].split(' ') for pair in labels_infos]

for p in labels_pairs:
    gt[int(p[0])] = [int(p[1])]

with open(bds_file, 'r') as f:
    bds_infos = f.readlines()
    bds_pairs = [pair[:-1].split(' ') for pair in bds_infos]

for p in bds_pairs:
    gt[int(p[0])].append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])

with open(save_gt_path, 'wb') as fw:
    pickle.dump(gt, fw, pickle.HIGHEST_PROTOCOL)