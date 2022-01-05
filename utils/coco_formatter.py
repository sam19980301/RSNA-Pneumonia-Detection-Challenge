import glob, pylab, pandas as pd
import pydicom, numpy as np
import os
from PIL import Image
import cv2
import json
from tqdm import tqdm
import random

kaggle_path = './rsna-pneumonia-detection-challenge/'

# if not os.path.exists('train_images'):
#     os.mkdir('train_images')
# if not os.path.exists('test_images'):
#     os.mkdir('test_images')

# print("save images into png format for COCO-dataset style")
# for file in tqdm(os.listdir(os.path.join(kaggle_path,'stage_2_train_images'))):
#     dcm_file = os.path.join(kaggle_path,'stage_2_train_images',file)
#     dcm_data = pydicom.read_file(dcm_file)
#     im = Image.fromarray(dcm_data.pixel_array)
#     im.save(os.path.join('./train_images',f'{file[:-4]}.png'))

# for file in tqdm(os.listdir(os.path.join(kaggle_path,'stage_2_test_images'))):
#     dcm_file = os.path.join(kaggle_path,'stage_2_test_images',file)
#     dcm_data = pydicom.read_file(dcm_file)
#     im = Image.fromarray(dcm_data.pixel_array)
#     im.save(os.path.join('./test_images',f'{file[:-4]}.png'))


images_list = os.listdir('train_images')
df = pd.read_csv(os.path.join(kaggle_path,'stage_2_train_labels.csv'))
# images_list = [f'{i}.png' for i in df[~df.isna().any(axis=1)]['patientId'].unique()]
random.seed(2021)
random.shuffle(images_list)
# train_images_list = images_list[:int(len(images_list)*0.95)]
# val_images_list = images_list[int(len(images_list)*0.95):]
# train_images_list = images_list[:-1000]
# val_images_list = images_list[-1000:]

def generate_ann(img_list,file_name):
    # Generate Coco Images Field
    # print("generate coco images field")
    images = list()
    for idx,img_name in tqdm(enumerate(img_list)):
        height, width = cv2.imread(os.path.join('train_images', img_name)).shape[:2]
        images.append(dict(
            id=idx,
            file_name=img_name,
            height=height,
            width=width))
    # Generate Coco Annotations Field
    # print("generate coco annotations field")
    obj_count = 0
    annotations = list()
    image_id_dict_ = {i['file_name'][:-4]:i['id'] for i in images}
    for idx in tqdm(range(len(df))):
        element = df.iloc[idx]
        if element.isna().any() or f"{element.patientId}.png" not in img_list:
            continue
        [x,y,width,height] = [element['x'],element['y'],element['width'],element['height']]
        label = int(element['Target']) 
        poly = [(x, y), (x + width, y),
                (x + width, y + height), (x, y + height)]
        poly = [float(p) for x in poly for p in x]
        data_anno = dict(
            image_id=image_id_dict_[element['patientId']],
            id=obj_count,
            category_id=label,
            bbox=[x, y, width, height],
            area=height * width,
            segmentation=[poly],
            iscrowd=0)
        annotations.append(data_anno)
        obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 1, 'name':'1'}]
    )
    with open(file_name,'w') as f:
        json.dump(coco_format_json,f)


# k-fold
k=5
for i in range(k):
    generate_ann([img for e,img in enumerate(images_list) if e%k!=i],f'RSNA_train_{i}.json')
    generate_ann([img for e,img in enumerate(images_list) if e%k==i],f'RSNA_val_{i}.json')