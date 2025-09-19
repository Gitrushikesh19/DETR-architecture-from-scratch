import os
import torch
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image


def load_images_and_labels(img_sets, label2idx, label_fname, split):
    img_infos = []
    for img_set in img_sets:
        img_names = []
        # Fetch all image names in txt file for this imageset
        for line in open(os.path.join(
                img_set, 'ImageSets', 'Main', '{}.txt'.format(label_fname))):
            img_names.append(line.strip())

        # Set annotation and image path
        label_dir = os.path.join(img_set, 'Annotations')
        img_dir = os.path.join(img_set, 'JPEGImages')

        for img_name in img_names:
            label_file = os.path.join(label_dir, '{}.xml'.format(img_name))
            img_info = {}
            label_info = ET.parse(label_file)
            root = label_info.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_info['img_id'] = os.path.basename(label_file).split('.xml')[0]
            img_info['filename'] = os.path.join(
                img_dir, '{}.jpg'.format(img_info['img_id'])
            )
            img_info['width'] = width
            img_info['height'] = height
            detections = []
            for obj in label_info.findall('object'):
                det = {}
                label = label2idx[obj.find('name').text]
                difficult = int(obj.find('difficult').text)
                bbox_info = obj.find('bndbox')
                bbox = [
                    int(bbox_info.find('xmin').text) - 1,
                    int(bbox_info.find('ymin').text) - 1,
                    int(bbox_info.find('xmax').text) - 1,
                    int(bbox_info.find('ymax').text) - 1
                ]
                det['label'] = label
                det['bbox'] = bbox
                det['difficult'] = difficult
                detections.append(det)
            img_info['detections'] = detections
            # Because we are using 25 as num_queries, as we have to ignore some of the images
            if len(detections) <= 25:
                img_infos.append(img_info)
    print('Total {} images found'.format(len(img_infos)))
    return img_infos


class CustomDataset(Dataset):
    def __init__(self, split, img_sets, img_size=640):
        self.split = split

        self.img_sets = img_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
        self.img_size = img_size
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        # Train and test transformations
        self.transforms = {
            'train': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.v2.RandomZoomOut(fill=self.img_mean),
                torchvision.transforms.v2.RandomIoUCrop(),
                torchvision.transforms.v2.RandomPhotometricDistort(),
                torchvision.transforms.v2.Resize(size=(self.img_size, self.img_size)),
                torchvision.transforms.v2.SanitizeBoundingBoxes(
                    labels_getter=lambda transform_input:
                    (transform_input[1]["labels"], transform_input[1]["difficult"])),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)

            ]),
            'test': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.Resize(size=(self.img_size, self.img_size)),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)
            ]),
        }

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        # We need to add background class as well with 0 index
        classes = ['background'] + classes

        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_label(self.img_sets,
                                                self.label2idx,
                                                self.fname,
                                                self.split)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        img_info = self.images_info[index]
        img = read_image(img_info['filename'])

        # Get annotations for this image
        targets = {}
        targets['boxes'] = tv_tensors.BoundingBoxes(
            [detection['bbox'] for detection in img_info['detections']],
            format='XYXY', canvas_size=img.shape[-2:])
        targets['labels'] = torch.as_tensor(
            [detection['label'] for detection in img_info['detections']])
        targets['difficult'] = torch.as_tensor(
            [detection['difficult']for detection in img_info['detections']])

        # Transform the image and targets
        transformed_info = self.transforms[self.split](img, targets)
        img_tensor, targets = transformed_info

        h, w = img_tensor.shape[-2:]

        # Boxes returned are in x1y1x2y2 format normalized from 0-1
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets['boxes'])
        targets['boxes'] = targets['boxes'] / wh_tensor
        return img_tensor, targets, img_info['filename']
