'''
Input image type: PIL.image, i.e. 'PIL.JpegImagePlugin.JpegImageFile'

'''

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
<<<<<<< HEAD:src/old_infer.py
import cv2
=======
import infer
>>>>>>> b6f1f163353510e5708bf53aff4cb93efc3a95c8:src/infer.py

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
parser.add_argument('-a','--arch', metavar = 'arch', default = 'yolov5n', help = model_list)
parser.add_argument('-i','--image-size', metavar = 'image_size', default= 224)
parser.add_argument('-d','--device', metavar = 'device', default = 'cuda')


def transform(image, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return transform(image)

<<<<<<< HEAD:src/old_infer.py
def work(image, ext_args, verbose = False):  #
=======
def work(image, ext_args):  #
>>>>>>> b6f1f163353510e5708bf53aff4cb93efc3a95c8:src/infer.py
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
<<<<<<< HEAD:src/old_infer.py
=======
    print(ext_args)
>>>>>>> b6f1f163353510e5708bf53aff4cb93efc3a95c8:src/infer.py
    for key, value in ext_args.items():  # update the args with external args
        vars(args)[key] = value
    print(args)
    model_name = args.arch
    device = args.device
    latency = 0
    work_start = time.time()

    ## process device
    assert device in ['cuda', 'cpu'], 'Device shall be "cuda" or "cpu" !'
    assert not ((not torch.cuda.is_available()) and (device == 'cuda')), 'set device of cuda, while it is not available'

    ## process image
    if args.image_size:
        image_size = (args.image_size, args.image_size)
    else:
        image_size =  image.size  # (w,h)
    data = image  # input data in type of PIL.image
    if model_name in cnn_model_list or model_name in deeplab_model_list:
        data = transform(image, image_size)
        data = torch.unsqueeze(data, dim =0)  # add a batch_size dimension

    ## process model
    assert model_name in model_list, f'Input model is not supported, shall be one of {model_list}'
    if model_name in cnn_model_list:
        model_func = 'models.' + model_name
        model = eval(model_func)(pretrained=True) # eval(): transform string to variable or function
    elif model_name in yolo_model_list:
<<<<<<< HEAD:src/old_infer.py
        device_yolo ='cpu'  if device == 'cpu' else 0
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device=device_yolo)
=======
        if device != 'cpu': device = 0
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device=device)
>>>>>>> b6f1f163353510e5708bf53aff4cb93efc3a95c8:src/infer.py

    ## process inference
    model.to(device)
    model.eval()
    with torch.no_grad():
        if device == 'cpu':
            start = time.time()
        else:  # cuda
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()  # cuda

        if model_name in cnn_model_list:  # cnn models
            data = data.to(device)
            prd = model.forward(data)
            prd = prd.to('cpu').detach()
<<<<<<< HEAD:src/old_infer.py
            result = np.argmax(prd, axis= 1).numpy()[0]  # transfer result from a tensor to a number

        elif model_name in yolo_model_list:  # YOLO models
            prd = model(data)
            result = prd.xyxyn

            # ## show result
            # if verbose:
            #     frame = np.squeeze(prd.render())
            #     cv2.imshow('Window_', frame)
            #     cv2.waitKey(200)
=======
            prd = np.argmax(prd, axis= 1).numpy()[0]  # transfer result from a tensor to a number

        elif model_name in yolo_model_list:  # YOLO models
            prd = model(data)
            ### show result
            # frame = np.squeeze(results.render())
            # cv2.imshow('Window_', frame)
            # if cv2.waitKey(800) & 0xFF >=0: break
>>>>>>> b6f1f163353510e5708bf53aff4cb93efc3a95c8:src/infer.py

        ## count latency
        if device == 'cpu':
            latency = (time.time() - start) * 1000  # metrics in ms
        else:  # gpu
            ender.record()
            torch.cuda.synchronize()  ###
            latency = starter.elapsed_time(ender)  # metrics in ms

    ## save log
    # t0= time.time()
    data_in_row = [work_start, args.arch, image_size, device, args.file_name, latency]
    logger_prefix = 'infer_log_'
    logger_by_date(data_in_row,args.log_dir, logger_prefix)
    # print('log overhead: ', time.time()-t0)

    return result, latency


if __name__ == '__main__':

    poisson = np.random.poisson(30, 4)
    # print(poisson)

    image_folder = '/home/royliu/Documents/datasets/temp/fold'
    i= 0
<<<<<<< HEAD:src/old_infer.py
    ext_args = dict(arch='yolov5s', device='cuda', image_size=224, verbose= True)
    for i, file_name in enumerate(os.listdir(image_folder)):
        ext_args['file_name']= file_name
        image = Image.open(os.path.join(image_folder, file_name))
        # image.show()
        result, latency = work(image,ext_args, verbose =True)
        print(f'Result: {result}, Latency: {latency}')
=======
    ext_args = dict(arch='alexnet', device='cuda', image_size=224)
    for i, file_name in enumerate(os.listdir(image_folder)):
        image = Image.open(os.path.join(image_folder, file_name))
        result, latency = work(image,ext_args)
        print(latency)
>>>>>>> b6f1f163353510e5708bf53aff4cb93efc3a95c8:src/infer.py
        if i >5: break  ## just test 5 images