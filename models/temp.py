# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.models.segmentation
# import argparse
# from tqdm import tqdm
# import sys
# sys.path.append("..")
# import datasets.cityscapes as cityscapes
#
# parser =  argparse.ArgumentParser()
# parser.add_argument('--root', metavar = 'root', default= '/home/royliu/Documents/datasets')
# parser.add_argument('--dataset', metavar = 'data_dir', default= 'cityscapes')
# parser.add_argument('--arch', metavar = 'arch', default = 'segmentation.deeplabv3_resnet50', help ='e.g. segmentation.deeplabv3_resnet101')
# args = parser.parse_args()
#
# def train_model(model,
#                 train_loader,
#                 val_loader,
#                 num_epochs,
#                 device):
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     lr = 0.001
#     device = device
#     # model.eval().to(device)
#     model.train().to(device)
#
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     for epoch in range(1, num_epochs + 1):
#         tr_loss = []
#         val_loss = []
#         print('Epoch {}/{}'.format(epoch, num_epochs))
#
#         for img, masks in tqdm(train_loader):
#             # inputs = torch.tensor(img).to(device)
#             # masks = torch.tensor(masks).to(device)
#             inputs = img.to(device)
#             masks = masks.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             # y_pred = outputs['out']
#             # y_true = masks
#             # loss = criterion(y_pred.float(), y_true.float())
#             loss = criterion(outputs['out'], masks)
#             loss.backward()
#             tr_loss.append(loss)
#             optimizer.step()
#             # break
#
#         print(f'Train loss: {torch.mean(torch.Tensor(tr_loss))}')
#
#         # for sample in tqdm(val_loader):
#         #     if sample['image'].shape[0] == 1:
#         #         break
#         #     inputs = sample['image'].to(device)
#         #     masks = sample['mask'].to(device)
#         for img, masks in tqdm(val_loader):
#             inputs = img.to(device)
#             masks = masks.to(device)
#
#             with torch.no_grad():
#                 outputs = model(inputs)
#             y_pred = outputs['out']
#             y_true = masks
#             loss = criterion(y_pred.float(), y_true.long())
#             val_loss.append(loss)
#             optimizer.step()
#             # break
#
#         print(f'Validation loss: {torch.mean(torch.Tensor(val_loss))}')
#
#     return model
#
# if __name__ == '__main__':
#     root = os.path.join(args.root, args.dataset)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print('Device:', device)
#     # model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=19)
#     model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=19)
#     train_loader = torch.utils.data.DataLoader(cityscapes.DataGenerator(root, split='train'),\
#                                                batch_size=1, num_workers=2)
#     val_loader = torch.utils.data.DataLoader(cityscapes.DataGenerator(root, split='val'), \
#                                              batch_size=1, num_workers=2)
#     epoch = 1
#     train_model(model, train_loader, val_loader, epoch, device)
#
