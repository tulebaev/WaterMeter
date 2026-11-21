"""
## Imports
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
import os
import math
import time
import warnings

from PIL import Image, ImageOps

"""
## Initialization
"""
warnings.filterwarnings('ignore')

IMG_MAXSIZE = 128
DIGITS_CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

DIGITS_NUMBER_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_', '№']

TYPES_CLASSES = ['BETAR', 'DECAST', 'ECONOM',
                 'GERRIDA', 'ITELMA', 'METER',
                 'NORMA', 'OTHERS', 'PULSAR',
                 'PULSE', 'VALTEC']
SIDES_CLASSES = ['OK', 'ROTATED']


"""
## Models Pathes
"""

METER_WEIGHTS_PATH = './final_models/model_meter_500.pth'
YOLO_PATH = './yolov5'
YOLO_NUMBER_DIGITS_PATH = './final_models/yolo_number_digits.pt'
MNIST_NUMBER_WEIGHTS_PATH = './final_models/digit_number_0.969.pth'
YOLO_DIGITS_PATH = './final_models/model_yolo_digits.pt'
MNIST_WEIGHTS_PATH = './final_models/model_mnist_ef.pth'
YOLO_CIRCLE_PATH = './final_models/model_yolo_circles.pt'
TYPE_WEIGHTS_PATH = "./final_models/model_type_ef.pth"
SIDE_WEIGHTS_PATH = "./final_models/model_side_ef_1.000.pth"
YOLO_METERS_PATH = './final_models/model_yolo_meters_number.pt'

"""
## Torch Functions
"""
start_transform = transforms.Compose([transforms.ToTensor()]) # transforms.Resize(SIZE),        
noise_transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
type_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize((128, 128)),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ])
side_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize((128, 128)),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ])

digits_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((64, 64)),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])

def resize_to_min(transform_data):
    data_shape = transform_data.shape
    if min(data_shape[1:]) > IMG_MAXSIZE:
        if min(data_shape[1:]) == data_shape[1]:
            test_size = (IMG_MAXSIZE, int((IMG_MAXSIZE/data_shape[1]) * data_shape[2]))
        else:
            test_size = (int((IMG_MAXSIZE/data_shape[2]) * data_shape[1]), IMG_MAXSIZE)
        resize_transform = transforms.Compose([transforms.Resize(test_size)])
        transform_data = resize_transform(transform_data)
    return transform_data

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

class DCDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((64, 64)),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = Image.fromarray(np.uint8(self.data[index])).convert('RGB')
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)

# %%
"""
## Initialize models
"""

# %%
# Meter_model
meter_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
meter_model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)) 
meter_model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1)) 
meter_model.load_state_dict(torch.load(METER_WEIGHTS_PATH))
meter_model.eval();

# Type_model
type_model = EfficientNet.from_name('efficientnet-b0')
type_model._fc = nn.Linear(1280, 11)
type_model.load_state_dict(torch.load(TYPE_WEIGHTS_PATH))
type_model.eval();

# Side_model
side_model = EfficientNet.from_name('efficientnet-b0')
side_model._fc = nn.Linear(1280, 2)
side_model.load_state_dict(torch.load(SIDE_WEIGHTS_PATH))
side_model.eval();

# Digit_model
classification_model = EfficientNet.from_name('efficientnet-b0')
classification_model._fc = nn.Linear(1280, 10)
classification_model.load_state_dict(torch.load(MNIST_WEIGHTS_PATH))
classification_model.eval();

# Digit_number_model
classification_number_model = EfficientNet.from_name('efficientnet-b0')
classification_number_model._fc = nn.Linear(1280, 12)
classification_number_model.load_state_dict(torch.load(MNIST_NUMBER_WEIGHTS_PATH))
classification_number_model.eval();

# %%
yolo_circles_model = torch.hub.load(YOLO_PATH, 'custom', path=YOLO_CIRCLE_PATH, source='local')

# %%
yolo_meters_model = torch.hub.load(YOLO_PATH, 'custom', path=YOLO_METERS_PATH, source='local')

# %%
yolo_digits_model = torch.hub.load(YOLO_PATH, 'custom', path=YOLO_DIGITS_PATH, source='local')

# %%
yolo_number_digits_model = torch.hub.load(YOLO_PATH, 'custom', path=YOLO_NUMBER_DIGITS_PATH, source='local')

def find_circle_yolo(cv_image):
    """
    ## Find_circle_yolo
    """
    # Prepare data:
    digits_meters_uint8 = cv_image.astype('uint8')

    # Predict
    results = yolo_circles_model(digits_meters_uint8, size=128)
    crops = results.crop(save=False)
    cropped_imgs = []

    for crop in crops:
        if crop['conf'].item() > 0.75:
            cropped_imgs.append(crop['im'])
    return cropped_imgs

def find_meters_yolo(cv_image):
    """
    ## Find_meters_yolo
    """
    #Prepare data:
    digits_meters_uint8 = cv_image.astype('uint8')

    #Predict
    results = yolo_meters_model(digits_meters_uint8, size=640)
    crops = results.crop(save=False)
    
    conf_left_meter = []
    crops_sort_meter = []
    conf_left_number = []
    crops_sort_number = []
    
    for crop in crops:
        if crop['cls'].item() == 0:
            conf_left_meter.append(crop['conf'].item())
        else:
            if crop['conf'] > 0.7:
                conf_left_number.append(crop['conf'].item())
            
    conf_left_meter.sort()
    conf_left_number.sort()
    
    for i in range(len(conf_left_meter)):
        for crop in crops:
            if crop['conf'].item() == conf_left_meter[i]:
                crops_sort_meter.append(crop)
    
    for i in range(len(conf_left_number)):
        for crop in crops:
            if crop['conf'].item() == conf_left_number[i]:
                crops_sort_number.append(crop)
                
    cropped_img_meter = crops_sort_meter[-1]['im']
    
    if len(crops_sort_number) > 0:
        cropped_img_number = crops_sort_number[-1]['im']
        cropped_box_number = np.array([x.item() for x in crops_sort_number[-1]['box']])
    else:
        cropped_img_number = None
        cropped_box_number = None
    return cropped_img_meter, cropped_img_number, cropped_box_number

def crop_circle(cv_image, out_circle):
    """
    ## Crop_circle
    """
    img = cv_image 
    mask = np.transpose(np.concatenate((out_circle, out_circle, out_circle), axis = 0), (1, 2, 0)).astype(np.float32)
    if img.shape[0] != mask.shape[0]:
        mask = np.transpose(mask, (1, 0, 2))
        mask = mask[:,::-1,:]
    _,threshold = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) ,60,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

def find_meters(cropped_img):
    """
    ## Find meters
    """
    # %%
    #Prepare img
    img = np.array(cropped_img)
    transform_data = start_transform(img)
    data_shape = transform_data.shape
    transform_data = resize_to_min(transform_data)
    if data_shape[0] == 1:
        transform_data = torch.cat((transform_data, transform_data, transform_data), dim=0)
    transform_data = noise_transform(transform_data) 

    #Predict
    outputs = meter_model(transform_data[None, ...])
    outputs = outputs['out']
    pred = torch.argmax(outputs, dim=1)
    resize_transform = transforms.Compose([transforms.Resize(data_shape[1:])])
    pred_r = resize_transform(pred)
    out = np.array(pred_r.detach())*255
    out_mask = out
    out = out[0][:,:,None]

    #Find angle and rotate
    thresh_image = out.astype(np.uint8)
    contours0, hierarchy = cv2.findContours(thresh_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours0) == 0:
        return None, None, None, None
    cnt = max(contours0, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
    edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
    usedEdge = edge1
    if cv2.norm(edge2) > cv2.norm(edge1):
        usedEdge = edge2
    reference = (1,0)
    angle = 180.0/math.pi * math.acos((reference[0]*usedEdge[0] + reference[1]*usedEdge[1]) / (cv2.norm(reference) *cv2.norm(usedEdge)))
    if usedEdge[1] < 0:
        angle = -angle
    rot_image = rotate_image(img, angle)
    img_float32 = np.float32(rot_image)
    rot_image = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR).astype(int)

    #Perspective transform
    if (usedEdge == edge1).all():
        pts1 = np.float32([box[0],box[1],box[3],box[2]])
    else:
        pts1 = np.float32([box[1],box[2],box[0],box[3]])

    pts2 = np.float32([[0, 0], [500, 0], [0, 100], [500, 100]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (550, 100))
    img_float32 = np.float32(result)
    digits_meters = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR).astype(int)
    
    return rot_image, digits_meters, out_mask, angle

def type_classification(rot_image):
    type_input = type_transforms(np.uint8(rot_image))
    outputs = type_model(type_input[None, ...])
    _, predicted = torch.max(outputs, 1)
    meter_type = TYPES_CLASSES[predicted[0]]
    return meter_type

def side_classification(rot_image):
    """
    ## Type classification
    """
    side_input = type_transforms(np.uint8(rot_image))
    
    outputs = side_model(side_input[None, ...])
    _, predicted = torch.max(outputs, 1)
    meter_side = SIDES_CLASSES[predicted[0]]
    return meter_side, (torch.max(outputs)-torch.min(outputs))/torch.abs(torch.max(outputs))
    
    
def find_digits(digits_meters):
    """
    ## YOlO Digits
    """
    
    # %%
    #Prepare data:
    digits_meters_uint8 = digits_meters.astype('uint8')

    #Predict
    results = yolo_digits_model(digits_meters_uint8, size=640)
    crops = results.crop(save=False)

    #Sort by x
    boxes_left = []
    crops_sort = []
    for crop in crops:
        boxes_left.append(crop['box'][0])
    boxes_left.sort()
    for i in range(len(boxes_left)):
        for crop in crops:
            if crop['box'][0] == boxes_left[i] and crop['conf'].item() > 0.6 and crop['im'].shape[0]/crop['im'].shape[1] < 2.5:
                crops_sort.append(crop)

    #Get Images
    digit_images = [crop['im'] for crop in crops_sort]
    confs = [crop['conf'] for crop in crops_sort]

    return digit_images, confs

def find_number_digits(digits_number):
    """
    ## YOlO Number Digits
    """
    
    # %%
    #Prepare data:
    digits_number_uint8 = digits_number.astype('uint8')

    #Predict
    results = yolo_number_digits_model(digits_number_uint8, size=640)
    crops = results.crop(save=False)

    #Sort by x
    boxes_left = []
    crops_sort = []
    for crop in crops:
        boxes_left.append(crop['box'][0])
    boxes_left.sort()
    for i in range(len(boxes_left)):
        for crop in crops:
            if crop['box'][0] == boxes_left[i] and crop['conf'].item() > 0.6 and crop['im'].shape[0]/crop['im'].shape[1] < 2.5:
                crops_sort.append(crop)

    #Get Images
    digit_images = [crop['im'] for crop in crops_sort]
    confs = [crop['conf'] for crop in crops_sort]

    return digit_images, confs

def digits_classification(images):
    """
    ## Digits Classification
    """
    dataset = DCDataset(images)
    dataloader = DataLoader(dataset, batch_size=len(images))

    batch = next(iter(dataloader))
    
    outputs = classification_model(batch)
    _, predicted = torch.max(outputs, 1)
    meters_value = ''.join(DIGITS_CLASSES[predicted[j]]
                                  for j in range(len(batch)))
    return meters_value

def digits_number_classification(images):
    """
    ## Digits Number Classification
    """
    dataset = DCDataset(images)
    dataloader = DataLoader(dataset, batch_size=len(images))

    batch = next(iter(dataloader))
    
    outputs = classification_number_model(batch)
    _, predicted = torch.max(outputs, 1)
    numbers_value = ''.join(DIGITS_NUMBER_CLASSES[predicted[j]]
                                  for j in range(len(batch)))
    return numbers_value

def resize_to(image, size = 640):
    rate = size/max(image.shape)
    width = int(image.shape[1] * rate)
    height = int(image.shape[0] * rate)
    dim = (width, height)
    resized = cv2.resize(image, dim)
    return resized
    
def read_image_from_file(image_path):
    pip_image = Image.open(image_path)
    pip_image = ImageOps.exif_transpose(pip_image)
    cv_image = cv2.imread(image_path)
    return pip_image, cv_image
    
def read_meters(cv_image):
    cropped_imgs = find_circle_yolo(cv_image)
    meter_types = []
    meters_values = []
    meter_sides = []
    value_numbers = []
    arrays_coords = []

    for cropped_img in cropped_imgs:
        rot_image, _, _, _ = find_meters(cropped_img)
        if rot_image is None: break
        meter_side, prob = side_classification(rot_image)
        if meter_side != 'OK':
            rot_image = cv2.rotate(rot_image, cv2.ROTATE_180)
        meter_type = type_classification(rot_image)

        rot_image = np.float32(rot_image)
        rot_image = cv2.cvtColor(rot_image, cv2.COLOR_BGR2RGB)

        rot_image = resize_to(rot_image, size = 640)
        digits_meters, image_number, cropped_box_number = find_meters_yolo(rot_image)

        if image_number is None:
            value_number = 'empty'
            array_coords = [0,0,0,0]
        else:
            array_coords = [coord.item() for coord in cropped_box_number]
            image_number = resize_to(image_number, size = 640)
            digit_number_images, confs = find_number_digits(image_number)
            value_number = digits_number_classification(digit_number_images)

        digits_meters = resize_to(digits_meters, size = 640)
        digit_images, _ = find_digits(digits_meters)
        meters_value = digits_classification(digit_images)

        meter_types.append(meter_type)
        meters_values.append(meters_value)
        meter_sides.append(meter_side)
        value_numbers.append(value_number)
        arrays_coords.append(array_coords)

    return meter_types, meters_values, meter_sides, value_numbers, arrays_coords


if __name__ == "__main__": 
    import sys
    if len(sys.argv) > 1:
        pip_image, cv_image = read_image_from_file(sys.argv[1])
        start = time.time()
        meter_type, meters_value, _, _, _ = read_meters(cv_image)
        elapsed = time.time() - start
        print(meter_type, meters_value)
        print('EXEC_TIME:', elapsed)
