import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np

labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
          "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
root = '/home/littlebee/dataset/VOC/'


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(in_file, classes):
    with open(in_file, 'r') as f:
        tree = ET.parse(f)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    file_name = root.find('filename').text
    result = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # if difficult:
        #     continue
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        result.append([cls_id, bb[0], bb[1], bb[2], bb[3]])
    return file_name, result


def main():
    ann_dir = root + 'VOC2007/Annotations/'
    traintxt = root + 'VOC2007/ImageSets/Main/train.txt'
    with open(traintxt, 'r') as f:
        imglist = f.readlines()
    annos = {}
    for imgid in imglist:
        annfile = ann_dir + imgid.rstrip() + '.xml'
        imgname, res = convert_annotation(annfile, labels)
        if len(res) == 0: continue
        annos[imgname] = res
    with open(root + 'annotations_train2007.pkl', 'wb') as f:
        f.write(pickle.dumps(annos))


if __name__ == '__main__':
    main()
