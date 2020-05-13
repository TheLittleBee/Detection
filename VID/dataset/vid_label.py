import xml.etree.ElementTree as ET
import pickle
import os
import numpy as np

codes = ["n02691156", "n02419796", "n02131653", "n02834778", "n01503061",
         "n02924116", "n02958343", "n02402425", "n02084071", "n02121808",
         "n02503517", "n02118333", "n02510455", "n02342885", "n02374451",
         "n02129165", "n01674464", "n02484322", "n03790512", "n02324045",
         "n02509815", "n02411705", "n01726692", "n02355227", "n02129604",
         "n04468005", "n01662784", "n04530566", "n02062744", "n02391049"]
labels = ["airplane", "antelope", "bear", "bicycle", "bird", "bus", "car",
          "cattle", "dog", "domestic_cat", "elephant", "fox", "giant_panda",
          "hamster", "horse", "lion", "lizard", "monkey", "motorcycle", "rabbit",
          "red_panda", "sheep", "snake", "squirrel", "tiger", "train", "turtle",
          "watercraft", "whale", "zebra"]
root = '/home/littlebee/dataset/VID/ILSVRC2015'


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
    result = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        trackid = int(obj.find('trackid').text)
        result.append([cls_id, bb[0], bb[1], bb[2], bb[3], trackid])
    return result


def main():
    train = True
    if train:
        ann_dir = os.path.join(root, 'Annotations/VID/train')
        out_file = os.path.join(root, 'annotations_train.pkl')
    else:
        ann_dir = os.path.join(root, 'Annotations/VID/val')
        out_file = os.path.join(root, 'annotations_val.pkl')
    annos = {}
    for d in os.listdir(ann_dir):
        if train:
            for d1 in os.listdir(os.path.join(ann_dir, d)):
                folder = os.path.join(d, d1)
                annos[folder] = []
                i = 0
                while os.path.exists(os.path.join(ann_dir, folder, '{:0>6}.xml'.format(i))):
                    annfile = os.path.join(ann_dir, folder, '{:0>6}.xml'.format(i))
                    annos[folder].append(convert_annotation(annfile, codes))
                    i += 1
        else:
            folder = d
            annos[folder] = []
            i = 0
            while os.path.exists(os.path.join(ann_dir, folder, '{:0>6}.xml'.format(i))):
                annfile = os.path.join(ann_dir, folder, '{:0>6}.xml'.format(i))
                annos[folder].append(convert_annotation(annfile, codes))
                i += 1
    with open(out_file, 'wb') as f:
        f.write(pickle.dumps(annos))


if __name__ == '__main__':
    main()
