import pickle
import socket
import os
import time
import numpy as np
from util import dict2str

server_adrr = ('127.0.0.1', 8000)

def send(dir, data_format, args, interval_list):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(server_adrr)
    args_str = dict2str(args)
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
            latency_list.append((now-start)*1000)
            start = now

            p = 0

            while p < data_len:
                to_send = data[p:p + data_len // 2]
                s.send(to_send)
                p += len(to_send)
            msg = pickle.loads(s.recv(1024))  # deserialize the result from server
            print('Result:', msg)

            # time.sleep(interval_list[i])  ## wait for a while,poisson interval
            if i < len(file_list)-1:  # i from 0 to n-1
                s.send(b'continue')

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
    np.random.seed(3)
    poisson = np.random.poisson(2, 3)
    interval_list = []
    for p in poisson:
        for i in range(p):
            interval_list.append(1 / p)  # in second metric. e.g. 0.16667
    print(interval_list[:])

    img_path = r'/home/royliu/Documents/datasets/temp/fold'
    args = dict(arch='yolov5s', device='cuda', image_size=224)
    start = time.time()
    send(img_path, 'jpg', args, interval_list)
    #

    # print("Time elapse for each image: ", (time.time() - start), 'sec.')
