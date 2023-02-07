'''

This seperate server and infer code supports image associates with different image.
'''
from PIL import Image
import io
import socket
import old_infer
from util import str2dict
import time
import numpy as np
import re
import cv2
import pickle


ip , port = '127.0.0.1', 8000
def work():
    s = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port))
    s.listen(3)
    reply =''
    while True:
        conn, addr = s.accept()
        print(f'Received request from {addr} !')
        header = conn.recv(1024)
        try:
            format, args_str = header.decode().split('|')
            args = str2dict(args_str)
            conn.send(b'ok')
        except:
            print('Fail to connect !')
            conn.close()
            continue
        msg = pickle.dumps('start....')
        while True:
            # print('checking checking msg from client...', reply)
            if reply == 'continuedone'  or reply =='done':  # msg to end
                print('Job from client is done ! \n Server is still waiting .....')
                break
            elif reply == 'continue' or reply =='':   # normal condition
                file_info = conn.recv(1024).decode()
            else:  ## file_len|file_name or continuefile_len|file_name
                file_info = re.findall(r'([0-9].+)', reply)[0]  ## parse the "continue" msg, which for avoiding blocking


            data_len, file_name = file_info.split('|')
            args['file_name'] = file_name

            conn.send(msg)
            if data_len and file_name:
                print(file_name)
                # newfile = open(os.path.join(dir, file_name), 'wb')  # save file transfered from server
                file = b''
                data_len = int(data_len)
                get = 0
                while get < data_len: # recieve data
                    data = conn.recv(data_len//2)
                    file += data  ## binary code
                    get += len(data)
                # conn.send(b'ok')
                print(f' {data_len} bytes to transfer, recieved {len(file)} bytes.')
                if file:  # file is in format of binary byte
                    pass
                    # newfile.write(file[:])  # save file transfered from server
                    # newfile.close()  # save file transfered from server

                    image = Image.open(io.BytesIO(file))  # convert binary bytes to PIL image in RAM

                    ## show image transfered
                    # cv_image = np.array(image)
                    # cv_image = cv_image[:, :, ::-1].copy()
                    # cv2.imshow('image', cv_image)
                    # cv2.waitKey(500)

                    ## predict image
                    prd, latency = infer.work(image, args)
                    msg = pickle.dumps(prd)  # serialize the result for sending back to client
                    conn.send(msg)
                    print(f' Result: {prd}, Latency: {latency} ms.')

            reply = conn.recv(1024).decode()
            # if reply == 'done':
            #     print('Job from client is done !')
            #     break
work()