import numpy as np
from my_dataset import MiniImageNet
from torch.utils.data import DataLoader
from torchvision import transforms, models
import time
import torch
import train
import multiprocessing as mp

# img_size = int(args.image_size)  # e.g. 224

def work(config, pipe, queue):
    # time.sleep(80)  # train starts after 20 sec. so set it another 20 sec. delay after traning
    con_1, con_2 = pipe
    queue.put(dict(process = 'infer'))  # for queue exception of this process
    model_name= config['arch']
    batch_size = config['batch_size']
    img_size = config['image_size']
    device = config['device']
    if not 'verbose' in config:
        config['verbose'] = None  ## this program is avoked by main_.py

    transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),  #224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(img_size+32), #256
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(img_size+32),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    root = '/home/royliu/Documents/dataset'
    model_func = 'models.'+ model_name
    data_test = MiniImageNet(root, 'test', transform)
    dataload_test = DataLoader(data_test, batch_size= batch_size, shuffle=True, num_workers=2)
    model = eval(model_func)(pretrained=True) # eval(): transform string to variable or function
    assert not ((not torch.cuda.is_available()) and (device =='cuda')), 'set device of cuda, while it is not available'

    model.to(device)
    model.eval()
    latency_total, latency, count = 0, 0, 0
    latency_list = []
    while True:  ## set for recive signal from training end
        ######### Below is for stoping infer after receiving msg from train.py ############
        con_2.close()
        quit_while = False
        with torch.no_grad():
            print()
            print('Inference starts......... ', config)
            # time.sleep(10)  # sleep 50 waiting for trainin
            for data, label in dataload_test:
                if device == 'cpu':
                    start = time.time()
                else: #cuda
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()  #cuda
                data = data.to(device)
                prd = model.forward(data)
                prd = prd.to('cpu').detach()
                # prd= prd.numpy()
                # prd = np.argmax(prd, axis=1)  # output is class_value

                ## count latency
                if device == 'cpu':
                    duration = (time.time() - start) *1000 # metrics in ms
                else:  # gpu
                    ender.record()
                    torch.cuda.synchronize()  ###
                    duration = starter.elapsed_time(ender) # metrics in ms
                latency_list.append(duration)
                count += 1

                target = label
                # measure accuracy and record loss
                acc1, acc5 = train.accuracy(prd, target, topk=(1, 5))
                if config['verbose'] == True:
                    print('acc1: ', acc1.numpy()[0]/100, ',        acc5: ', acc5.numpy()[0]/100)  # convert tensor to numpy

                try:  # if train end, then terminate infer here
                    if con_1.poll() and config['verbose'] != True:  # verbose to control if this program avoke locally (infer.py)
                        msg = con_1.recv()  # receive msg from train
                        if msg == 'done':  # signal from training process to end, ROy added on 12092022
                            queue.put(dict(latency=latency))  # queue.put(dict(process = 'infer')), {'latency': latency}
                            con_1.close()
                            print()
                            # print("herrrrrrr Latency is:", latency, ' ms.')
                            print('Infer is terminated after train stop!')
                            quit_while = True   # so, it will quit both for loop and while loop
                            break
                except Exception as e:
                    print(e)
                    break
            if config['verbose'] == True:  # means the program is initiate her,i.e. infer.py, so quite after one epoch
                quit_while = True
                print('Infering done! Total: ', len(dataload_test), 'interations!')
            if quit_while == True:
                break
            ### ^^^^^^Above is for stoping infer after receiving msg from train.py^^^^^^^^^^ ####
    latency= np.mean(latency_list[3:])  # take off GPU warmup
    print('Inferenc latency is: ', latency/batch_size, 'ms. Total ', count*batch_size,' images are infered.' )
    print('acc1: ', acc1.numpy()[0]/100, ',        acc5: ', acc5.numpy()[0]/100)  # convert tensor to numpy


if __name__ == '__main__':

    config ={'arch': 'resnet50','workers': 1, 'epochs': 10, 'batch_size': 32, 'image_size':224, 'device':'cuda', 'verbose': True}
    pipe3, pipe4 = mp.Pipe()
    queue = mp.Queue()
    pipe =(pipe3, pipe4)
    ## note:  loop of inference start from here is infinete since no train.py stop it
    ## set verbose true here, otherwise Pipe will make the infer only take once (of course, I can add one more variable to control, but I didn't want).
    work(config, pipe, queue)

