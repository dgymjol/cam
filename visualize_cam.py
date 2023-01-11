import os
import numpy as np
import random
import torch
import argparse
from model.ResNet50_cam import ResNet50_cam
import cv2
from PIL import Image
import torchvision.transforms as transforms


def init_seed(seed):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def get_parser():
    parser = argparse.ArgumentParser(
        description='Visualization of CAM')
    
    parser.add_argument(
            '--weights',
            default='work_dir/cub/r50cam_0.001:92')

    parser.add_argument(
            '--image-id',
            # type=int,
            default='1')

    parser.add_argument(
            '--results-dir',
            default='results_dir/cub')

    return parser

def IoU(box1, box2):
    '''
        box1 : (x, y, w, h)
        box2 : (x, y, w, h)
    '''

    # box = (x1, y1, x2, y2)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2]) # x + w
    y2 = min(box1[1] + box1[3], box2[1] + box2[3]) # y + h

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    iou = inter / (box1_area + box2_area - inter + 1e-7)
    return iou

# argpase
init_seed(2023)
parser = get_parser()
arg = parser.parse_args()


# mkdir - save dir
if not os.path.isdir(arg.results_dir):
    os.makedirs(arg.results_dir)

# load model & weights
exp_dir, epoch = arg.weights.split(':')
if not os.path.exists(exp_dir):
    raise Exception(f"{exp_dir} : No such file or directory")
else:
    model_weight_file_name = ''
    for run_file in os.listdir(exp_dir) :
        if f"runs-{epoch}" in run_file  and '.pt' in run_file:
            model_weight_file_name = run_file
    if model_weight_file_name == '':
        raise Exception(f'that epoch {epoch} weight file doesnt exist')

    weights = torch.load(os.path.join(exp_dir,model_weight_file_name))
    model = ResNet50_cam(num_classes=200)
    model.load_state_dict(weights, strict=False)
    print("pretrained weight is loaded")

# load dataset infos
data_path = 'data/CUB_200_2011'
train_test_split_file = os.path.join(data_path, 'train_test_split.txt')
image_names_file = os.path.join(data_path, 'images.txt')
class_names_file = os.path.join(data_path, 'classes.txt')
labels_file = os.path.join(data_path, 'image_class_labels.txt')
bds_file = os.path.join(data_path, 'bounding_boxes.txt')
images_path = os.path.join(data_path, 'images')
save_train_data = os.path.join(data_path, 'train_data.pkl')
save_test_data = os.path.join(data_path, 'test_data.pkl')

## print test or train image
indices = np.loadtxt(train_test_split_file)
for index in indices:
    if str(index[0]) == arg.image_id:
        if index[1]: 
            print("train_image")
        else:
            print("test_image")
        break

# find path corresponding to the id
with open(image_names_file, 'r') as f:
    image_infos = f.readlines()
    for pair in image_infos:
        id, path = pair[:-1].split(' ')
        if id == arg.image_id:
            print(f'image_path : {path}')
            image_path = os.path.join(data_path, 'images',path)
            break

# find label corresponding to the id
with open(labels_file, 'r') as f:
    label_infos = f.readlines()
    for pair in label_infos:
        id, cls = pair[:-1].split(' ')
        if id == arg.image_id:
            print(f'gt_cls : {cls}')
            break

# find bounding box corresponding to the id
bbox_data = dict()
with open(bds_file, 'r') as f:
    bds_infos = f.readlines()
    for pair in bds_infos:
        id, x, y, w, h = pair[:-1].split(' ')
        if id == arg.image_id:
            gt_bbox = [float(x), float(y), float(w), float(h)]
            # print(f'image_gt_bbox : {gt_bbox}')
            break


test_image = Image.open(image_path)
test_image_cv = cv2.imread(image_path)

w, h = test_image.size

with torch.no_grad():
    transform_eval = transforms.Compose([ transforms.Resize(int(448/0.875)),
                                        transforms.CenterCrop(448),
                                        transforms.ToTensor(),
                                        ])     

    normalize = transforms.Normalize( mean=(0.485, 0.456, 0.406),
                                            std=(0.229, 0.224, 0.225))

    t = normalize(transform_eval(test_image)).cuda() # (3, 324, 500)
    model = model.cuda()
    model.eval()
    conf, gaps = model(t.unsqueeze(0)) # [1, 200] [1, 200, 24 ,32]
    _, pred_label = torch.max(conf, 1)
    pred_label = int(pred_label)

    print("pred_label : ", pred_label)

gap = gaps[0][pred_label] # (24, 32)
gap = gap - torch.min(gap)
gap = gap / torch.max(gap)
gap = gap.detach().cpu().numpy()
gap_image = np.uint8(255*gap)

heatmap = cv2.applyColorMap(cv2.resize(gap_image,(w, h)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + test_image_cv * 0.5

cv2.imwrite(f'{arg.results_dir}/{arg.image_id}_heatmap.jpg', result)

cam_image = cv2.resize(gap_image,(w, h))
threshold = np.max(cam_image) * 0.70

    # threshold map 
# _, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_BINARY)
    # or (this one : Otsu's binarization method 적용한 것)
_, thresh_map = cv2.threshold(cam_image, threshold, 255, cv2.THRESH_OTSU)


"""
If you want to generate threshold map : below
"""
thresh_map_rgb = np.zeros_like(test_image_cv)

thresh_map_rgb[:, :, 0] = thresh_map
thresh_map_rgb[:, :, 1] = thresh_map
thresh_map_rgb[:, :, 2] = thresh_map

result = thresh_map_rgb * 0.3 + test_image_cv * 0.5

"""
generating bounding box and evaluate IOU
"""

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_map)

largest_connected_component_idx = np.argmax(stats[1:, -1]) + 1 # background is most
x, y, w, h = stats[largest_connected_component_idx][:-1] #(x, y, width, height)
gt_x, gt_y, gt_w, gt_h = gt_bbox
iou = IoU(gt_bbox, [x, y, w, h])
print(f"IoU : {iou}")

cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), thickness = 1)
cv2.rectangle(result, (int(gt_x), int(gt_y)), (int(gt_x)+int(gt_w), int(gt_y)+int(gt_h)), (255, 0, 0), thickness = 1)
cv2.putText(result, "IoU : {:.4f}".format(iou), (10, 20),cv2.FONT_ITALIC, 0.5, (200,200,200), 1)

cv2.imwrite(f'{arg.results_dir}/{arg.image_id}_bbox.jpg', result)