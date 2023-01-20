'''
// https://cocodataset.org/#download

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

//wget http://images.cocodataset.org/zips/test2017.zip
//wget http://images.cocodataset.org/zips/unlabeled2017.zip

91 classes, 80 classes for object detection

YOLO txt label downloads:  https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip ,
        or convert scropt: https://github.com/ultralytics/JSON2YOLO

data structure (YOLO format):
  coco
    ├─labels
    │   ├─ train2017.txt
    │   ├─ val2017.txt
    │   └─ test-dev2017.txt
    └─Images
         ├─ train2017
         │     ├─  000000000???.jpg
         │     ├─  000000000???.jpg
         │     └─  ...
         ├─ val2017
         │     ├─  000000000???.jpg
         │     ├─  000000000???.jpg
         │     └─  ...
         └─ test2017
               ├─  000000000???.jpg
               ├─  000000000???.jpg
               └─  ...

'''

import pandas as pd
import os
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--root', metavar= 'root', default = '/home/royliu/Documents/datasets/coco')
parser.add_argument('--split', metavar='split', default='test', help='"train" or "val" or "test".')


class test_dataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # if there is any arg or kw, inherit!
        args = parser.parse_args()
        self.root = os.path.join(args.root, './images/test2017')
        self.images = []
        for file_name  in os.listdir(self.root):
            self.images.append(os.path.join(self.root, file_name))

    def _transform(self, image_pil):
        transform = transforms.Compose([
                                transforms.Resize((256, 512)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return transform(image_pil)

    def __getitem__(self,index):
        image= self.images[index]
        # image= Image.open(self.images[index])
        # image = self._transform(image)
        return image

    def __len__(self):
        return len(self.images)

def main():
    test_loader = DataLoader(test_dataset(), batch_size=1, num_workers= 1, shuffle=False)


if __name__== '__main__':
    main()