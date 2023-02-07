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
import torch
from torchvision import transforms, models
from util import logger_by_date

ip , port = '127.0.0.1', 8000

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet50', 'deeplabv3_mobilenet_v3_large']
model_list = cnn_model_list + yolo_model_list + deeplab_model_list

def load_model(model_name, device):
    ## process model
    assert model_name in model_list, f'Input model is not supported, shall be one of {model_list}'
    if model_name in cnn_model_list:
        model_func = 'models.' + model_name
        model = eval(model_func)(pretrained=True) # eval(): transform string to variable or function
    elif model_name in yolo_model_list:
        device_yolo ='cpu'  if device == 'cpu' else 0
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device=device_yolo)
    return model

def transform(image, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return transform(image)

def work():
    s = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port))
    s.listen(3)
    reply =''
    previous_model = ''
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
                print('Job from client is done ! \n Server is waiting .....')
                break
            elif reply == 'continue' or reply =='':   # normal condition
                file_info = conn.recv(1024).decode()
            else:  ## file_len|file_name or continuefile_len|file_name
                file_info = re.findall(r'([0-9].+)', reply)[0]  ## parse the "continue" msg, which for avoiding blocking


            data_len, file_name = file_info.split('|')
            args['file_name'] = file_name
            model_name, device = args['arch'], args['device']

            if model_name !=previous_model :  # if the model is the same as previous one , no need load model again
                model=load_model(model_name, device)
                previous_model = model_name

            ## process device
            assert device in ['cuda', 'cpu'], 'Device shall be "cuda" or "cpu" !'
            assert not ((not torch.cuda.is_available()) and (
                        device == 'cuda')), 'set device of cuda, while it is not available'



            conn.send(msg)
            if data_len and file_name:
                work_start = time.time()
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

                    ## process image
                    image = Image.open(io.BytesIO(file))  # convert binary bytes to PIL image in RAM
                    if args['image_size']:  # w/h
                        image_size = (args['image_size'], args['image_size'])
                    else:
                        image_size = image.size  # (w,h)
                    data = image  # input data in type of PIL.image
                    if model_name in cnn_model_list or model_name in deeplab_model_list:
                        data = transform(image, image_size)
                        data = torch.unsqueeze(data, dim=0)  # add a batch_size dimension

                    ## show image transfered
                    # cv_image = np.array(image)
                    # cv_image = cv_image[:, :, ::-1].copy()
                    # cv2.imshow('image', cv_image)
                    # cv2.waitKey(500)

                    ## predict image
                    ## process inference

                    model.to(device)
                    model.eval()
                    with torch.no_grad():
                        if device == 'cpu':
                            start = time.time()
                        else:  # cuda
                            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                                enable_timing=True)
                            starter.record()  # cuda

                        if model_name in cnn_model_list:  # cnn models
                            data = data.to(device)
                            prd = model.forward(data)
                            prd = prd.to('cpu').detach()
                            prd = np.argmax(prd, axis=1).numpy()[0]  # transfer result from a tensor to a number

                        elif model_name in yolo_model_list:  # YOLO models
                            prd = model(data)
                            prd = prd.xyxyn

                        if device == 'cpu':
                            latency = (time.time() - start) * 1000  # metrics in ms
                        else:  # gpu
                            ender.record()
                            torch.cuda.synchronize()  ###
                            latency = starter.elapsed_time(ender)  # metrics in ms

                    # prd, latency = infer.work(image, args)
                    msg = pickle.dumps(prd)  # serialize the result for sending back to client
                    conn.send(msg)
                    print(f' Result: {prd}, Latency: {latency} ms.')

                ## save log
                # t0= time.time()
                data_in_row = [work_start, model_name, image_size, device, args['file_name'], latency]
                logger_prefix = 'infer_log_'
                logger_by_date(data_in_row, '../result/log', logger_prefix)
                # print('log overhead: ', time.time()-t0)

            reply = conn.recv(1024).decode()
            # if reply == 'done':
            #     print('Job from client is done !')
            #     break
work()