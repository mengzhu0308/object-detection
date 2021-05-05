#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/5/4 18:17
@File:          voc_annotation.py
'''

from xml.etree import ElementTree as ET

root_dir = 'D:/datasets'
sets = [('2012', 'train'), ('2012', 'val')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes2ids = {j: i for i, j in enumerate(classes)}

def convert_annotation(year, image_id):
    with open(f'{root_dir}/VOCdevkit/VOC{year}/Annotations/{image_id}.xml', 'r', encoding='utf-8') as f:
        tree = ET.parse(f)

    root = tree.getroot()

    image_info = []

    size = root.find('size')
    w, h = int(size.findtext('width')), int(size.findtext('height'))
    image_info.append(f'{w},{h}')

    for obj in root.iter('object'):
        difficult = obj.findtext('difficult')
        cls = obj.findtext('name')
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes2ids[cls]
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.findtext('xmin')), int(xmlbox.findtext('ymin')),
             int(xmlbox.findtext('xmax')), int(xmlbox.findtext('ymax')))

        image_info.append(f'{b[0]},{b[1]},{b[2]},{b[3]},{cls_id}')

    return image_info

for year, image_set in sets:
    with open(f'{root_dir}/VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    image_ids = [line.strip() for line in lines]

    write_text = []
    for image_id in image_ids:
        write_line = [f'{root_dir}/VOCdevkit/VOC{year}/JPEGImages/{image_id}.jpg']
        write_line.extend(convert_annotation(year, image_id))
        write_text.append(' '.join(write_line) + '\n')

    write_text[-1] = write_text[-1].strip()
    with open(f'model_data/voc{year}_{image_set}.txt', 'w', encoding='utf-8') as f:
        f.writelines(write_text)