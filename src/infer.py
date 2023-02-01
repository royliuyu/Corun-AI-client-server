import os
import sys
import numpy as np
sys.path.append('../models')
import argparse
import deeplab_v3
import torch
import time
import PIL.Image as Image
from torchvision import transforms, models
from util import logger_by_date

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet50', 'deeplabv3_mobilenet_v3_large']
model_list = cnn_model_list + yolo_model_list + deeplab_model_list

parser = argparse.ArgumentParser()
parser.add_argument('--root', metavar = 'root', default= '/home/royliu/Documents/datasets')
parser.add_argument('--log-dir', metavar = 'log_dir', default= '../result/log')
parser.add_argument('-a','--arch', metavar = 'arch', default = 'densenet121', help = model_list)
parser.add_argument('-i','--image-size', metavar = 'image_size', default= 224)
parser.add_argument('-d','--device', metavar = 'device', default = 'cuda')


def transform(image, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return transform(image)

def work(image_path):  #
    '''
    input:
        image path and file name
        model name (mendarory)
        imange size (optional)
        device type (optional)

    output:
        prediction
        latency
    '''
    args =  parser.parse_args()
    model_name = args.arch
    device = args.device
    latency = 0
    work_start = time.time()

    ## process device
    assert device in ['cuda', 'cpu'], 'Device shall be "cuda" or "cpu" !'
    if model_name in yolo_model_list  and device == 'cuda':  device = 0

    ## process image
    image = Image.open(image_path)
    if args.image_size:
        image_size = (args.image_size, args.image_size)
    else:
        image_size =  image.size  # (w,h)
    data = transform(image, image_size)
    data = torch.unsqueeze(data, dim =0)  # add a batch_size dimension

    ## process model
    assert model_name in model_list, f'Input model is not supported, shall be one of {model_list}'
    model_func = 'models.' + model_name
    model = eval(model_func)(pretrained=True) # eval(): transform string to variable or function
    assert not ((not torch.cuda.is_available()) and (device =='cuda')), 'set device of cuda, while it is not available'

    ## process inference
    model.to(device)
    model.eval()
    with torch.no_grad():
        if device == 'cpu':
            start = time.time()
        else:  # cuda
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()  # cuda

        data = data.to(device)
        prd = model.forward(data)
        prd = prd.to('cpu').detach()

        ## count latency
        if device == 'cpu':
            latency = (time.time() - start) * 1000  # metrics in ms
        else:  # gpu
            ender.record()
            torch.cuda.synchronize()  ###
            latency = starter.elapsed_time(ender)  # metrics in ms

    if model_name in cnn_model_list:
        prd = np.argmax(prd, axis= 1).numpy()[0]  # transfer result from a tensor to a number
    elif model_name in yolo_model_list:
        pass

    ## save log
    # t0= time.time()
    data_in_row = [work_start, args.arch, image_size, image_path, latency]
    logger_prefix = 'infer_log_'
    logger_by_date(data_in_row,args.log_dir, logger_prefix)
    # print('log overhead: ', time.time()-t0)

    return prd, latency


if __name__ == '__main__':
    image_path = '/home/royliu/Documents/datasets/mini_imagenet/images_temp/n0153282900000019.jpg'
    # result, latency = main(image_path)
    # print(f'Result is {result}, latency is {latency} ms.')

    poisson = np.random.poisson(30, 60)
    # print(poisson)

    image_folder = '/home/royliu/Documents/datasets/coco/images/test2017'
    i= 0
    for i, image in enumerate(os.listdir(image_folder)):
        result, latency = work(os.path.join(image_folder, image))
        print(latency)
        if i >5: break