# 和多个客户端通讯 - client端：
import socket
import time

import pandas as pd
import time
import os
import asyncio
import pickle


#
# # 1. test for trasfer words
# df = pd.DataFrame()
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect(('127.0.0.1', 8001))
# word_list = 15*['Abundance', 'Accomplish', 'Adversity', 'Aesthetic', 'Affection', 'Ambition', 'Amicable', 'Amplify', 'Appreciation', 'Authentic', 'Benevolent', 'Blissful', 'Boldness', 'Bountiful', 'Brilliant', 'Capability', 'Captivating', 'Carefree', 'Celebrate', 'Champion', 'Charisma', 'Cheerful', 'Clarity', 'Coherence', 'Collaboration', 'Comfortable', 'Compassionate', 'Composure', 'Concentration', 'Confidence', 'Congruent', 'Consciousness', 'Consistency', 'Contentment', 'Conviction', 'Courageous', 'Creativity', 'Cultivate', 'Curiosity', 'Daring', 'Dazzling', 'Decisive', 'Delightful', 'Determined', 'Devotion', 'Dignity', 'Discipline', 'Discovery', 'Diversity', 'Eager', 'Ease', 'Ecstatic', 'Education', 'Efficiency', 'Elegance', 'Elevate', 'Empathy', 'Empower', 'Endurance', 'Energy', 'Enlighten', 'Enthusiasm', 'Equal', 'Equilibrium', 'Essence', 'Esteem', 'Excellence', 'Excitement', 'Exhilaration', 'Experience', 'Exploration', 'Expression', 'Extend', 'Extraordinary', 'Eyesight', 'Fabulous', 'Fairness', 'Faith', 'Famous', 'Fascinating', 'Fearless', 'Felicity', 'Fidelity', 'Flourishing', 'Focus', 'Fortitude', 'Freedom', 'Friendship', 'Fulfillment', 'Generosity', 'Genius', 'Genuine', 'Gladness', 'Glorious', 'Graceful', 'Gratitude', 'Greatness']
# word_list = word_list + ['exit']
#
# for word in word_list:
#     start = time.time()
#     re_data = word
#     client.send(re_data.encode("utf-8"))
#     if 'exit' == re_data:
#         print('客户端退出..')
#         break
#     data = client.recv(1024)
#     res = data.decode('utf-8')
#     # df.append(word)
#     print(res)
# client.close()



## 2. test for transfer images
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


async def handle(dir, file_name):
    # await asyncio.sleep(0.02)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 10002))

    s.send(b'start')
    data_format = 'jpg'
    header = data_format + '|' + 'hello'
    s.send(header.encode())
    reply = s.recv(1024)

    if reply.decode() == 'ok':
        start = time.time()
        ## 1. Send image to server
        data = open_file(os.path.join(dir, file_name))
        data_len = len(data)
        file_info = str(data_len) + '|' + file_name
        s.send(file_info.encode())
        msg = s.recv(1024)  # for unblocking
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
    # return result_cont_size, result_from_server, start, time.time()

async def task_coro(dir, file_list):
    tasks = []  ## tasks for coroutine running
    # results = await asyncio.gather(tasks)
    accum_interval = 0
    for i in range(200):
        if i >= len(file_list) - 1: break  # if out of file_list, quit
        file_name = file_list[i]
        tasks.append(asyncio.ensure_future(handle(dir, file_name)))
    await asyncio.wait(tasks)
    # return asyncio.wait(tasks)

root = os.environ['HOME']
dir = os.path.join(root, r'./Documents/datasets/coco/images/test2017')
file_list = os.listdir(dir)
file_list.sort(key=lambda x: x[:-4])  # take out surfix (.cvs)


loop = asyncio.get_event_loop()
# loop.run_until_complete(asyncio.wait(tasks))
loop.run_until_complete(task_coro( dir, file_list))
loop.close()

