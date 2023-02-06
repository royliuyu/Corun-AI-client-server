import socket
import os
import time
import numpy as np
from util import dict2str

server_adrr = ('127.0.0.1', 8000)

def send(dir, data_format, args):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(server_adrr)
    args_str = dict2str(args)
    header = data_format + '|' + args_str
    s.send(header.encode())
    reply = s.recv(1024)
    file_list = os.listdir(dir)
    file_list.sort(key=lambda x: x[:-4])
    if reply.decode() == 'ok':
        for i, file_name in enumerate(file_list):
            data = open_file(os.path.join(dir, file_name))
            data_len = len(data)
            file_info = str(data_len) + '|' + file_name
            s.send(file_info.encode())
            s.recv(1024)

            p = 0
            while p < data_len:
                to_send = data[p:p + data_len // 2]
                s.send(to_send)
                p += len(to_send)
            s.recv(1024).decode()
            if i < len(file_list)-1:  # i from 0 to n-1
                s.send(b'continue')
        s.send(b'done')
        s.close()


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
    poisson = np.random.poisson(10, 6)
    interval = []
    print(poisson)
    for p in poisson:
        for i in range(p):
            interval.append(1 / p)
    for itv in interval:
        img_path = r'/home/royliu/Documents/datasets/temp/fold'
        args = dict(arch='alexnet', device='cuda', image_size=224)
        start = time.time()
        send(img_path, 'jpg', args)
        time.sleep(itv)

    print("Time elapse for each image: ", (time.time() - start), 'sec.')
