# 和多个客户端通讯 - SERVER端：
import socket
import threading
import pandas as pd
import time
import os
from util import logger_by_date, date_time
import re
from PIL import Image
import io
import socket
from util import str2dict, date_time
import time
import numpy as np
import pickle
import asyncio
#
# # 1. test for trafering word
# def deal(conn, client):
#     df = pd.DataFrame(columns= ['a','b'])
#     print(f'新线程开始处理客户端 {client} 的请求数据')
#
#     while True:
#         start = time.time()
#         data = conn.recv(1024).decode('utf-8')  # 接收客户端数据并且解码， 一次获取 1024b数据(1k)
#         # print('接收到客户端发送的信息：%s' % data)
#         if 'exit' == data:
#             print('客户端发送完毕，已断开连接')
#             break
#         re_data = data.upper()
#         conn.send(re_data.encode('UTF-8'))
#         # df.iloc[len(df)] =[start, re_data]
#         df = df.append({'a':start, 'b':re_data}, ignore_index=True)
#         dt, tm = date_time()
#         log_dir = os.path.join(os.environ['HOME'], r'./Documents/profile_train_infer/result/log/infer_server', dt)
#         if not os.path.exists(log_dir): os.makedirs(log_dir)
#         col = ['a', 'b']
#         data_in_row = [start, re_data]
#         logger_prefix = 'text'
#         logger_by_date(col, data_in_row, log_dir, logger_prefix)
#
#     conn.close()
#
#     return df
#
#
# # 类型：socket.AF_INET 可以理解为 IPV4
# # 协议：socket.SOCK_STREAM
# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind(('0.0.0.0', 8001))  # (client客户端ip, 端口)
# server.listen()  # 监听
#
# while True:
#     sock, addr = server.accept()  # 获得一个客户端的连接(阻塞式，只有客户端连接后，下面才执行)
#     # xd = threading.Thread(target=deal, args=(sock, addr))
#     # xd.start()  # 启动一个线程
#     df = deal(sock, addr)
#     # print(df)




## 2. test : image tranfered from client
print_interval = 1
async def handle(conn, addr): ## recv_send  between clinets, one turn of data transfer+ result back
    reply =''
    previous_model = ''
    print(f'Received request from {addr} !')
    log = pd.DataFrame()
    header = conn.recv(1024)
    ## 1. receive data from client
    try:
        format, args_str = header.decode().split('|')
        conn.send(b'ok')
    except:
        print('Fail to connect !')
        conn.close()
        return ##continue
    cnt = 0
    while True:
        # print('checking checking msg from client...', reply)
        if reply == 'continuedone' or reply == 'done':  # msg indicates task ends from client
            print(' Job from client is done!\n', '=' * 70, '\n' * 3, 'Server is listening.....')
            break
        elif reply == 'continue' or reply == '':  # normal condition, '' for the first processing
            file_info = conn.recv(1024).decode()
        else:  ## i.e. file_len|file_name or continuefile_len|file_name
            try:  ## solve that can't recognize "done" msg due to packets sticking issue
                file_info = re.findall(r'([0-9].+)', reply)[0]  ## parse the "continue" msg, which for avoiding blocking
            except:
                break

        try:  ## solve can't recognize "done" msg due to packets sticking issue, the last packets of file_info goes ahead of "continue"
            data_len, file_name = file_info.split('|')  ## parse header with file length
        except:
            print(' Job from client is done.\n', '=' * 70, '\n\n\nServer is listening.....')
            break

        msg = 'start to recieve transfered file from client'.encode()  ##
        conn.send(msg)
        if data_len and file_name:
            work_start = time.time()
            # newfile = open(os.path.join(dir, file_name), 'wb')  # save file transfered from server
            file = b''
            data_len = int(data_len)
            get = 0
            while get < data_len:  # recieve data
                data = conn.recv(data_len // 2)
                file += data  ## binary code
                get += len(data)

            if cnt % print_interval == 0:
                print(f' {cnt}: File name :{file_name}, {data_len} bytes to transfer, recieved {len(file)} bytes.')
            if file:  # file is in format of binary byte
                # newfile.write(file[:])  # save file transfered from server
                # newfile.close()  # save file transfered from server

                ## 1.2 process image
                # image = Image.open(io.BytesIO(
                #     file))  # convert binary bytes to PIL image in RAM, i.e. 'PIL.JpegImagePlugin.JpegImageFile'
                # if len(np.array(image).shape) < 3:  # gray image
                #     image = image.convert("RGB")  # if gray imange , change to RGB (3 channels)
                #

                # ## 1.3 process inference
                result, latency = 'yolo', 23

                ### 2. send result back the result to client

                result_cont = pickle.dumps({'file_name': file_name, 'latency_server(ms)': latency,
                                            'result': result})  # serialize the result for sending back to client
                result_cont_size = len(result_cont)

                conn.send(str(result_cont_size).encode())
                msg = conn.recv(1024).decode()

                p = 0
                while p < result_cont_size:
                    to_send = result_cont[p:p + result_cont_size // 2]
                    conn.send(to_send)
                    p += len(to_send)

                if cnt % print_interval == 0:
                    print(f' {cnt}: File name: {file_name}, Result: {result}, Latency: {latency} ms.\n')

            # save log
            dt, tm = date_time()  # catch current datatime
            col =['time', 'data']
            data_in_row = [work_start,  latency]
            print(data_in_row)
            # logger_prefix = 'test_server'
            # log_dir = os.path.join(os.environ['HOME'], r'./Documents/profile_train_infer/result/log/infer_server', dt)
            # if not os.path.exists(log_dir): os.makedirs(log_dir)
            # logger_by_date(col, data_in_row, log_dir, logger_prefix)

        reply = conn.recv(1024).decode()
        cnt += 1
    reply = conn.recv(1024).decode()  # to recieve notice when client starts a new task
    # return data_in_row, logger_prefix


async def main(s):
    while True:
        conn, addr = s.accept()
        await handle(conn, addr)
        # thd = threading.Thread(target = handle, args =(conn, addr))
        # thd.start()
        # thd.join()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # check port confilict and reconnect when fail
s.bind(('0.0.0.0', 10002))
s.listen(50)
asyncio.run(main(s))

