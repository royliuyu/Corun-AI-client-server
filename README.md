# Measuring the throughput and tail Latency of Concurrent Model Training and Inferences

## Bibtex
    This is the official code for paper "Corun: Concurrent Inference and Continuous Training at the Edge for Cost-Efficient AI-Based Mobile Image Sensing"
    Welcome to explore based on it and cite with:
    @article{liu2024corun,
      title={Corun: Concurrent Inference and Continuous Training at the Edge for Cost-Efficient AI-Based Mobile Image Sensing},
      author={Liu, Yu and Andhare, Anurag and Kang, Kyoung-Don},
      journal={Sensors},
      volume={24},
      number={16},
      pages={5262},
      year={2024},
      publisher={MDPI}
    }

##  Concurrently run multiple CNN models inference and training on edge server, where requests are sent from client. Server sends inference results back to client.
   - We are now supprting the models of:
     - cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                    'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                    'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                    'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
     - yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
     - deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large']
     - dehazing_model_list = ['RIDCP_dehazing']

## In "src" folder:
 - main.py
   - implement single training and single inference tasks to run concurrently
   - set training and inference co-running configure in config.json
 - profiler.py
   - run it for profiling GPU and CPU state, at a 1 second interval
   - hardware is set in device.json
 - multi_client.py and multi_server.py
   - Implement single training and multiple inference tasks to run concurrently: 
 - multi_infer.py
   - implement multiple inference tasks
 - client.py, client_asysnc.py, server.py
   - act as sub-functions called by above programs 
   - client.py spawn client in synchronized mode, while client_asysnc.py is in an asynchronized mode
## In "datasets" folder:
 - dataset and dataloader generators 
## In "models" folder:
 - build DNN models
## In "result" folder:
 - save logs including infomation of model names, timestamps, latencis, etc.

## Dataset（in client side)：
   - chck root directory in work() of client.py:
     - root = os.environ['HOME']
     - data_dir = os.path.join(root, r'./Documents/datasets/')
     - datasets are phsycally available in:
       - root+data_dir+'./coco/images/test2017'
       - root+data_dir+'./mini_imagenet/images'
       - root+data_dir+'./dehazing/test_224'
       
## Setup：
 - Option1. Corun with multiple inference and single training:
   Deploy these codes both on Server and Client. Multi_client sends inference request and setup train model.  
   - If want to run a train model concurrently with above multiple inference on server:
     - On client, multi_client.py: name the training model in variable: "train_model_name". It's for logging
     - On server: setup training config and in config.json, inference arch set 'none'
     - On server: Run main.py before running multi_client and multi_server
   - Server side, in main() of multi_server.py:
     - Setup ip and port variable
     - Setup server_num variable
     - Run multi_server.py, it will spawn multiple processes to call server.py
     - Experiment results are saved in "result/infer_server" folder   
   - Client side, in main() of multi_client.py:
     - Setup ip, port
     - Setup inference models in variable "arch_zoo"
     - Setup train models in variable "train_model_name", if no train , setup "none"
     - Setup request_rate_list variable, e.g. [0] stands for use one Azure's
     - Setup client_num_list variable, e.g. [4,3] stands for run 4 combinations and 3 combinations
     - Setup how may clients to run concurrently, in "client_num_list"
     - Run multi_client.py, it will spawn multiple processes to call client.py to send requests
     - Experiment results are saved in "result/infer_client" folder
   - execution time depends how the long the interval list is, if use poission, we setup 120 seconds.
     in the codes of "poisson = np.random.poisson(request_rate, 120)" in client.py
 - Option 2. Corun single training and single inference on server, client send inference request
   - Run server.py on server
   - Run client.py on client
 - Option 3. Profile CPU and GPU on server side, with main.py and profiler.py
   - Setup models in config.json (Decide if use any training or inference model run during profiling. Set "None" if not need.)
   - Setup profile number (1 instance/second) with variable profiling_num in main.py
   - Result will be saved in folder of result/log/$datatime

