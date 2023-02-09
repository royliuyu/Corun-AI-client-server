import pickle
import socket
import os
import time
import numpy as np
from util import dict2str, logger_by_date

ip , port = '128.226.119.73', 51400
# ip , port = '127.0.0.1', 51400

def send(dir, data_format, args, interval_list):
    work_start = time.time()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip , port))
    args_str = dict2str(args)
    s.send(b'start')
    header = data_format + '|' + args_str
    s.send(header.encode())
    reply = s.recv(1024)

    file_list = os.listdir(dir)
    file_list.sort(key=lambda x: x[:-4])
    latency_list =[]
    start = time.time()
    if reply.decode() == 'ok':
        for i, file_name in enumerate(file_list):
            if i>= len(interval_list):
                break # quit for loop when interval list is over
            data = open_file(os.path.join(dir, file_name))
            data_len = len(data)
            file_info = str(data_len) + '|' + file_name
            s.send(file_info.encode())
            s.recv(1024) # unblock
            # print('Result:', msg)
            now = time.time()
            # print('Latency:', (now-start)*1000 , 'ms.')
            latency = (now-start)*1000
            latency_list.append(latency)
            start = now

            p = 0

            while p < data_len:
                to_send = data[p:p + data_len // 2]
                s.send(to_send)
                p += len(to_send)
            msg = pickle.loads(s.recv(1024))  # deserialize the result from server
            print(f'Result from server: {msg}, latency {latency}. ')

            # time.sleep(interval_list[i])  ## wait for a while,poisson interval
            if i < len(file_list)-1:  # i from 0 to n-1
                s.send(b'continue')
            else :
                s.send(b'done')

            # log the instance
            data_in_row = [work_start, args['arch'], args['image_size'], args['device'], file_name, latency]
            logger_prefix = 'infer_log_client_'
            logger_by_date(data_in_row, '../result/log', logger_prefix)

        s.send(b'done')
        s.close()
        print('Latency(ms): ', latency_list[:])



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

if __name__ == '__main__':
    np.random.seed(6)
    poisson = np.random.poisson(10, 600)
    # print(poisson)
    interval_list = []
    for p in poisson:
        for i in range(p):
            interval_list.append(1 / p)  # in second metric. e.g. 0.16667
    # print(interval_list[:])
    print(len(interval_list))

    # img_path = r'/home/lab/Documents/datasets/temp/fold'
    img_path = r'/home/lab/Documents/datasets/coco/images/test2017'
    # args = dict(arch='deeplabv3_resnet50', device='cuda', image_size=224)  #deeplabv3_resnet50
    # start = time.time()
    # send(img_path, 'jpg', args, interval_list)
    # time.sleep(1)
    # print("Time elapse for each image: ", (time.time() - start), 'sec.')

    arch_list = cnn_model_list = ['alexnet',  'densenet121',  'efficientnet_v2_l', \
                  'googlenet', 'inception_v3',  'mobilenet_v3_small',  'resnet50',  \
                                  'vgg16', 'yolov5s', 'deeplabv3_resnet50']

    for arch in arch_list:
        args = dict(arch=arch, device='cuda', image_size=224)  # deeplabv3_resnet50
        start = time.time()
        send(img_path, 'jpg', args, interval_list)
        time.sleep(1)
