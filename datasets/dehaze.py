import os
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', metavar= 'data-dir', default = './Documents/datasets/dehazing')
parser.add_argument('--split', metavar='split', default='test', help='"train" or "val" or "test".')
root = os.environ['HOME']


class test_dataset(Dataset):
    def __init__(self, image_size ,*args, **kwargs):
        super().__init__(*args, **kwargs) # if there is any arg or kw, inherit!
        args = parser.parse_args()
        self.data_dir = os.path.join(root, args.data_dir, './test_224')
        self.images = []
        self.image_size = image_size
        for file_name  in os.listdir(self.data_dir):
            self.images.append(os.path.join(self.data_dir, file_name))

    def _transform(self, image_pil):
        transform = transforms.Compose([
                                transforms.Resize(self.image_size),
                                transforms.ToTensor(),
        ])
        return transform(image_pil)

    def __getitem__(self,index):
        # image= self.images[index]  # type of str
        image= Image.open(self.images[index])
        if len(np.array(image).shape) < 3:  # gray image
            image = image.convert("RGB")  # if gray imange , change to RGB (3 channels)
        image = self._transform(image)
        return image

    def __len__(self):
        return len(self.images)

def main():
    test_loader = DataLoader(test_dataset(image_size = (224,224)), batch_size=1, num_workers= 1, shuffle=False)
    print('Total image number:', len(test_loader))

if __name__== '__main__':
    main()