import asyncio
import pickle
import socket
import os
import time
import datetime
import numpy as np
from util import dict2str, logger_by_date, date_time
import threading

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large']

def open_file(file_path):
    data = b''
    try:
        file = open(file_path, 'rb')
        data = file.read()
    except:
        print(f' Error ! Get file path {file_path} ')
    else:
        file.close()
        return data

def image_folder(data_dir, model):
    if model in cnn_model_list:
        folder = './mini_imagenet/images'
        # folder = './coco/images/test2017'
    elif model in deeplab_model_list:
        folder = './google_street/part1'
        # folder = './coco/images/test2017'
    else:
        folder = './coco/images/test2017'
    return os.path.join(data_dir,folder)

def gen_poission_interval(request_rate):
    poisson = np.random.poisson(request_rate, 60)
    # print(poisson)
    interval_list = []
    for p in poisson:
        for i in range(p):
            interval_list.append(1 / p)  # in second metric. e.g. 0.16667
    print('Instance amount :', len(interval_list))
    return interval_list

async def handle(ip, port, dir, file_name, args, accum_interval):
    await asyncio.sleep(accum_interval)
    start = time.time()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))

    args_str = dict2str(args)
    data_format = 'jpg'
    header = data_format + '|' + args_str
    s.send(header.encode())
    reply = s.recv(1024)

    file_list = os.listdir(dir)
    file_list.sort(key=lambda x: x[:-4])  # take out surfix (.cvs)
    latency_list = []
    start = time.time()
    if reply.decode() == 'ok':
        ## 1. Send image to server
        data = open_file(os.path.join(dir, file_name))
        data_len = len(data)
        file_info = str(data_len) + '|' + file_name
        s.send(file_info.encode())
        msg = s.recv(1024)  # for unblocking,
        # print('recived file info:', msg.decode())
        p = 0
        while p < data_len:
            to_send = data[p:p + data_len // 2]
            s.send(to_send)
            p += len(to_send)

        ## 2. receive result from server
        # ## 2.1 option 1: receive result from server,  old version of sending small # of result value
        # result_from_server = pickle.loads(s.recv(1024))  # old: deserialize the result from server

        # ## 2.2 option 2: receive result from server
        result_cont_size = int(s.recv(1024).decode())
        s.send('Result size recieved'.encode())
        if result_cont_size > 0:
            result_cont = b''
            get = 0
            while get < result_cont_size:  # recieve data
                data = s.recv(result_cont_size // 2)
                result_cont += data  ## binary code
                get += len(data)
        result_from_server = pickle.loads(result_cont)
        s.send(b'continue')
    s.send(b'done')
    s.close()
    # return result_cont_size, result_from_server, start, time.time()

async def task_coro(ip, port, img_folder, args, request_interval_list, print_interval):
    dir = img_folder
    tasks = []  ## tasks for coroutine running
    # results = await asyncio.gather(tasks)
    accum_interval = 0
    file_list = os.listdir(dir)
    file_list.sort(key=lambda x: x[:-4])  # take out surfix (.cvs)
    for i in range(len(request_interval_list)):
        # if i >= len(file_list) - 1: break  # if out of file_list, quit
        if i >= 200: break  # if out of file_list, quit
        accum_interval += request_interval_list[i]
        file_name = file_list[i]
        tasks.append(asyncio.ensure_future(handle(ip, port, dir, file_name, args, accum_interval)))
    await asyncio.wait(tasks)

def work(ip, port, request_rate_list, arch_list, train_model_name, print_interval,*args):
    np.random.seed(6)
    print(f'Print status every {print_interval} records.')
    print('Start time: ', datetime.datetime.now())
    root = os.environ['HOME']
    data_dir = os.path.join(root, r'./Documents/datasets/')

    for request_rate in request_rate_list:
        if request_rate == 0:  ## means customized, not poisson
            request_interval_list = args[-1]  # additional arg for request_interval_list only
            print('Instance amount (customized) : ', len(request_interval_list))
        else:
            request_interval_list = gen_poission_interval(request_rate)

        for arch in arch_list:
            args = dict(request_rate = request_rate, arch=arch, train_model_name = train_model_name, device='cuda', image_size=224)  # deeplabv3_resnet50
            img_folder = os.path.join(root, r'./Documents/datasets/coco/images/test2017')
            # img_folder = image_folder(data_dir, args['arch'])  # select dataset to fit models

            # send(ip, port, img_folder, args, request_interval_list,print_interval)
            loop = asyncio.get_event_loop()
            # loop.run_until_complete(asyncio.wait(tasks))
            loop.run_until_complete(task_coro(ip, port, img_folder, args, \
                                              request_interval_list,print_interval))
            loop.close()
            time.sleep(1) # take a rest after one experiment finish


if __name__ == '__main__':
    ip , port = '192.168.85.73', 54100
    # ip, port = '127.0.0.1', 51401
    ip, port = '127.0.0.1', 54100
    # ip,port = '128.226.119.71', 51400
    print_interval = 10  # to change this value to change the result displaying frequency on the screen

    arch_list = [  'resnet50', 'alexnet','deeplabv3_resnet50',  'densenet121',\
                 'efficientnet_v2_l', 'googlenet', 'inception_v3+', 'mobilenet_v3_small','yolov5s', 'vgg16' ]
    arch_list = ['resnet50']
    ## train_model_list = ['none', 'resnet152_32', 'vgg16_64', 'deeplabv3_resnet50_8']
    train_model_name = 'none'  # manually change the name here , batch size as well!!
    request_rate_list = [40]  # if there contains 0, must follow a customized_request_interval_list argument in the end in work()
    customized_request_interval_list = [1,2,3,4]
    args = (ip, port, request_rate_list, arch_list, train_model_name, print_interval)
    ######################## run the inference combinations ########################
    work(ip, port, request_rate_list, arch_list, train_model_name, print_interval, customized_request_interval_list)
    ################################################################################
