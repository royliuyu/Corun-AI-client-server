'''
download code via: https://github.com/ultralytics/yolov5/tree/e83b422a69bbd69628687b2dc50102c08877505c

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# COCO 2017 dataset http://cocodataset.org by Microsoft
# Example usage: python cnn_train.py --data coco.yaml
# parent
# ├── yolov5
# └── datasets  # revise this if need, in yolov5/data/coco.yaml
#     └── coco  ← downloads here (20.1 GB)
            └── images
            └── labels
            └── val2017.txt
            └── weights
                    └── yolov5x.pt
            └── ...
https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml

refer: https://zhuanlan.zhihu.com/p/357701213

1.  add __init__.py under yolov5 folder
2. put the dataset under root folder,
        e.g. '/home/royliu/Documents'
3.  check and change(if need) yolov5/data/coco.yaml's datasets path
        e.g. path: ../dataset/coco
'''


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('../datasets')
import coco
import cv2
import numpy as np
import time

def work_infer(config, pipe, queue):
    queue.put(dict(process= 'yolo_v5'))  # for queue exception of this process
    con_yolo_a,con_yolo_b = pipe
    con_yolo_a.close()
    test_loader = DataLoader(coco.test_dataset(), batch_size= 1, num_workers= 1, shuffle=False)

    device = 0 if config['device'] == 'cuda' else 'cpu' #'cuda':0. or 'cpu'
    batch_size =  config['batch_size']

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device = device)  # yolov5n - yolov5x6 or custom
    model.conf = 0.6  # confidence threshold (0-1) 0.52
    model.iou = 0.5  # NMS IoU threshold (0-1). 0.45

    t_cpu, t_gpu = 0, 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy
    latency_total, latency, count = 0, 0, 0
    latency_list = []
    quit_while = False

    while True:
        # for i, (batches) in enumerate(tqdm(test_loader)):
        for i, (batches) in enumerate(test_loader):
            for image in batches:
                if device == 'cpu':
                    start = time.time()  # for cpu time counting
                else:
                    starter.record()  # for gpu time counting

                results = model(image)


                if device == 'cpu':
                    t_cpu = (time.time() - start) *1000  # in ms, for cpu exe time counting
                else:
                    ender.record()  # for gpu exe time counting
                    torch.cuda.synchronize()  ###
                    t_gpu = starter.elapsed_time(ender)   ### 计算时间

                # cv2.imshow('Result', np.squeeze(results.render()))
                # cv2.waitKey(100)
                # cv2.destroyAllWindows()

                if device == 0:
                    latency_list.append(t_gpu)
                else:
                    latency_list.append(t_cpu)

                count += 1

            try:  # if train end, then terminate infer here
                # if con_inf_b.poll() and config['verbose'] != True:  # verbose to control if this program avoke locally (cnn_infer.py)
                if con_yolo_b.poll() :
                    msg = con_yolo_b.recv()  # receive msg from train
                    if msg == 'stop':  # signal from training process to end, ROy added on 12092022
                        print('Infer: get "stop" notice from main. ')
                        con_yolo_b.close()
                        latency = np.mean(latency_list[3:])  # take off GPU warmup
                        print('Infer: Quiting..., average latency is %0.5f ms. Total %d instances are infered.'% (latency,count*batch_size))
                        queue.put(dict(latency=latency))  # queue.put(dict(process = 'infer')), {'latency': latency}
                        quit_while = True   # so, it will quit both for loop and while loop

                        break
            except Exception as e:
                print(e)
                break

        if quit_while == True:
            break

    print('Inferenc latency is: ', latency / batch_size, 'ms. Total ', count * batch_size,
          ' images are infered.')
            # if device == 'cpu':
            #     latency = np.mean(exe_time_cpu[2:])
            # else:
            #     latency = np.mean(exe_time_gpu[2:])
            #
            # print(latency)

if __name__ == '__main__':
    work_infer()