import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation
import multiprocessing as mp
import argparse
from tqdm import tqdm
import sys
sys.path.append("../datasets")
import cityscapes


parser =  argparse.ArgumentParser()
parser.add_argument('--root', metavar = 'root', default= '/home/royliu/Documents/datasets')
parser.add_argument('--dataset', metavar = 'data_dir', default= 'cityscapes')
parser.add_argument('--arch', metavar = 'arch', default = 'segmentation.deeplabv3_resnet50', help ='e.g. segmentation.deeplabv3_resnet101')
args = parser.parse_args()


def work_train(config, queue):
    ## initialize the arguments
    queue.put(dict(process='deeplabv3_train'))  # for queue exception of this process
    args = parser.parse_args()
    model_func = 'torchvision.models.'+args.arch # load parser_args's arch value before overiding by config
    # update argparse' args with the new_args from parent process (main)
    for key, value in config.items():
        vars(args)[key] = value  # update args
    print('Train starts .........', args)
    root = os.path.join(args.root, args.dataset)  # add cityscapes folder with root
    epoch = args.epochs
    batch_size = args.batch_size
    num_workers = args.workers
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device
    print('Device:', device)
    assert args.arch == 'deeplab_v3', f'Model \"{args.arch}\" is not recognized!'

    model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=19)
    # model = eval(model_func)(num_classes=19)

    train_loader = torch.utils.data.DataLoader(cityscapes.DataGenerator(root, split='train'), \
                                               batch_size= batch_size, num_workers= num_workers)
    val_loader = torch.utils.data.DataLoader(cityscapes.DataGenerator(root, split='val'), \
                                             batch_size= batch_size, num_workers= num_workers)

    train(model, train_loader, val_loader, epoch, device, batch_size)

def train(model, train_loader, val_loader, num_epochs, device, batch_size):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    device = device

    '''
    to solve exception when batch size =1, since Batch normalization computes: y = (x - mean(x)) / (std(x) + eps) = 0 then.
    call model.eval() to disable further training. This stops BatchNorm layers from updating their mean and variance
    '''
    if batch_size == 1:
        model.eval().to(device)
    else:
        model.train().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        tr_loss = []
        val_loss = []
        print('Epoch {}/{}'.format(epoch, num_epochs))

        for img, masks in tqdm(train_loader):
            # inputs = torch.tensor(img).to(device)
            # masks = torch.tensor(masks).to(device)
            inputs = img.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # y_pred = outputs['out']
            # y_true = masks
            # loss = criterion(y_pred.float(), y_true.float())
            loss = criterion(outputs['out'], masks)
            loss.backward()
            tr_loss.append(loss)
            optimizer.step()
            # break

        print(f'Train loss: {torch.mean(torch.Tensor(tr_loss))}')

        # for sample in tqdm(val_loader):
        #     if sample['image'].shape[0] == 1:
        #         break
        #     inputs = sample['image'].to(device)
        #     masks = sample['mask'].to(device)
        for img, masks in tqdm(val_loader):
            inputs = img.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            y_pred = outputs['out']
            y_true = masks
            loss = criterion(y_pred.float(), y_true.long())
            val_loss.append(loss)
            optimizer.step()
            # break

        print(f'Validation loss: {torch.mean(torch.Tensor(val_loss))}')

    return model

if __name__ == '__main__':
    queue = mp.Queue()
    config = {"arch": "deeplab_v3", "workers": 1, "epochs": 3, "batch_size":4,  "image_size": 224,  "device": "cuda"}
    work_train(config, queue)


