import torch
from torch.utils.data import DataLoader
import time
import multiprocessing as mp
import os
dehazing_model_list = ['RIDCP_dehazing']
root = os.environ['HOME']
import sys
sys.path.append('./RIDCP_dehazing')
sys.path.append('../datasets')
from basicsr.archs.dehaze_vq_weight_arch import VQWeightDehazeNet
from PIL import Image
import cv2
import numpy as np
import dehaze
from basicsr.utils import img2tensor, tensor2img


def work_infer(config, pipe, queue):
    queue.put(dict(process= 'dehazing'))  # for queue exception of this process
    con_yolo_a,con_yolo_b = pipe
    con_yolo_a.close()
    test_loader = DataLoader(dehaze.test_dataset((config['image_size'],config['image_size'])),\
                             batch_size= 1, num_workers= 1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size =  config['batch_size']
    assert  config['arch'] in dehazing_model_list, f'only sypport {dehazing_model_list} !'
    ## load dehazing model
    weight_path = 'RIDCP_dehazing/pretrained_models/pretrained_RIDCP.pth'
    # set up the model
    model = VQWeightDehazeNet(codebook_params=[[64, 1024, 512]], \
                                 LQ_stage=True, use_weight=False, weight_alpha=-21.25).to(device)
    model.load_state_dict(torch.load(weight_path)['params'], strict=False)

    t_cpu, t_gpu = 0, 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  ###Roy
    latency_total, latency, count = 0, 0, 0
    latency_list = []
    quit_while = False

    print("Infering starts:", config)
    while True:
        # for i, (batches) in enumerate(tqdm(test_loader)):
        for i, (batches) in enumerate(test_loader):
            # for img_tenser in batches:
            if device == 'cpu':
                start = time.time()  # for cpu time counting
            else:
                starter.record()  # for gpu time counting

            output, _ = model.test(batches.to(device))

            print(output)
            ### show the dehazed image
            # image = output[0].permute(1, 2, 0).cpu().numpy()
            # image = (image * 255).astype(np.uint8)
            # cv2.imshow('Dehazed Image 0', image)
            # if cv2.waitKey(0) & 0xFF >=0: break

            if device == 'cpu':
                t_cpu = (time.time() - start) *1000  # in ms, for cpu exe time counting
            else:
                ender.record()  # for gpu exe time counting
                torch.cuda.synchronize()  ###
                t_gpu = starter.elapsed_time(ender)   ### 计算时间

            if device == 0:
                latency_list.append(t_gpu)
            else:
                latency_list.append(t_cpu)

            count += len(batches)

            try:  # if train end, then terminate infer here
                # if con_inf_b.poll() and config['verbose'] != True:  # verbose to control if this program avoke locally (cnn_infer.py)
                if con_yolo_b.poll() :
                    msg = con_yolo_b.recv()  # receive msg from train
                    if msg == 'stop':  # signal from training process to end, ROy added on 12092022
                        print('Infer: get "stop" notice from main. ')
                        con_yolo_b.close()
                        latency = np.mean(latency_list[3:])  # take off GPU warmup
                        print('Infer: Quiting..., average latency is %0.5f ms. Total %d instances are infered.'% (latency,count*batch_size))
                        queue.put(dict(latency_ms=latency))  # queue.put(dict(process = 'infer')), {'latency': latency}
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
    config = {'arch': 'RIDCP_dehazing','workers': 1, 'epochs': 10, 'batch_size': 1, 'image_size':224, 'device':'cuda', 'verbose': True}
    pipe3, pipe4 = mp.Pipe()
    queue = mp.Queue()
    pipe =(pipe3, pipe4)
    ## note:  loop of inference start from here is infinete since no cnn_train.py stop it
    ## set verbose true here, otherwise Pipe will make the infer only take once (of course, I can add one more variable to control, but I didn't want).
    work_infer(config, pipe, queue)
