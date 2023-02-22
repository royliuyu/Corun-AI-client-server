

import pickle
import socket
import os
import time
import numpy as np
from util import dict2str, logger_by_date

# ip , port = '192.168.85.73', 51400
# ip , port = '127.0.0.1', 51400
ip , port = '128.226.119.71', 51400

print_interval = 1  # to change this value to change the result displaying frequency on the screen

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
            msg = s.recv(1024) # for unblocking
            # print('recived file info:', msg.decode())

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

            # ## option 1: receive result from server,  old version of sending small # of result value
            # result_from_server = pickle.loads(s.recv(1024))  # old: deserialize the result from server

            # ## option 2: receive result from server
            # s.send('Ready to recieve results size from server.'.encode())
            result_cont_size = s.recv(1024).decode()
            result_cont_size = int(result_cont_size)
            # print(result_cont_size)
            s.send('Result size recieved'.encode())
            if result_cont_size > 0:
                result_cont = b''
                get = 0
                while get < result_cont_size:  # recieve data
                    data = s.recv(result_cont_size // 2)
                    result_cont += data  ## binary code
                    get += len(data)
            result_from_server =  pickle.loads(result_cont)

            if i% print_interval == 0:
                print(f'{i}: Result data size: {result_cont_size} bytes, Result from server: {result_from_server}, latency {latency}. ')  # print this consume latency

            if i < len(file_list)-1:  # i from 0 to n-1
                s.send(b'continue')
            else :
                s.send(b'done')

            # log the instance
            col = ['work_start', 'infer_model_name', 'train_model_name','image_size', 'device', 'file_name', 'latency', 'request_rate']
            data_in_row = [work_start, args['arch'], args['train_model_name'], args['image_size'], args['device'], file_name, latency], args['request_rate']
            logger_prefix = 'infer_log_client_' + 'request_rate_' + str(args['request_rate']) + ' train_' + args[
                'train_model_name'] + ' infer_' + args['arch'] + '_'
            logger_by_date(col, data_in_row, '../result/log', logger_prefix)

        s.send(b'done')
        s.close()
        # print('Latency(ms): ', latency_list[:])

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
    request_rate_list = [20,40, 60]
    for request_rate in request_rate_list:
        poisson = np.random.poisson(request_rate, 600)
        # print(poisson)
        interval_list = []
        for p in poisson:
            for i in range(p):
                interval_list.append(1 / p)  # in second metric. e.g. 0.16667
        print('Instance amount :', len(interval_list))
        print(f'Print status every {print_interval } records.')

        # img_path = r'/home/lab/Documents/datasets/temp/fold'
        img_path = r'/home/anurag/Documents/coco/images/test2017'

        arch_list = cnn_model_list = ['yolov5s','alexnet',  'densenet121',  'efficientnet_v2_l', \
                      'googlenet', 'inception_v3',  'mobilenet_v3_small',  'resnet50',  \
                                      'vgg16',  'deeplabv3_resnet50']
        train_model_list = ['none','alexnet', 'vgg16',  'deeplabv3_resnet50']

        for arch in arch_list:
            train_model_name = 'none' #manually change the name here , batch size as well!!
            args = dict(request_rate = request_rate, arch=arch, train_model_name = train_model_name, device='cuda', image_size=224)  # deeplabv3_resnet50
            start = time.time()
            send(img_path, 'jpg', args, interval_list)
            time.sleep(1)