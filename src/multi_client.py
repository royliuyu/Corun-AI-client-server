from multiprocessing import Pool
import client
import os
import mp_exception as mp_new

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large']

def image_folder(data_dir, model):
    if model in cnn_model_list:
        # folder = './mini_imagenet/images'
        folder = './coco/images/test2017'
    elif model in deeplab_model_list:
        folder = './google_street/part1'
    else:
        folder = './coco/images/test2017'
    return os.path.join(data_dir,folder)

def main():
    basic_ip ='127.0.0.1'
    # basic_ip = '192.168.85.71'
    basic_port = 54101
    client_num = 5
    print_interval = 1000
    train_model_name = 'None'
    assert client_num < 9, f'server num. {client_num} is too big, no larger than 8.'  # support 9 servers at most
    addr_list = []
    [addr_list.append((basic_ip, port)) for port in range(basic_port, basic_port + client_num)]
    arch_list = ['yolov5s', 'alexnet', 'deeplabv3_resnet50','resnet50', 'vgg16']
    # data_dir = os.path.join(os.environ['HOME'], './Documents/datasets')
    request_rate_list = [10, 20, 40, 60] ## ms
    # pool = Pool(server_num)

    p_list = []
    for i, addr in enumerate(addr_list):
        print(f'{arch_list[i]} is runing')
        ip, port = addr
        try:
            ### call: work(ip, port, request_rate_list, arch_list, train_model_name, print_interval )
            args= (ip, port, request_rate_list , [arch_list[i]],train_model_name, print_interval,)
            # pool.apply_async(client.work, args=args)  # change to below for passing exceptions
            p_list.append(mp_new.Process(target=client.work, args=(args)))

        except:
            break
    for p in p_list: p.start()
    for p in p_list: p.join()
    for p in p_list: p.terminate()

    # pool.close()
    # pool.join()

if __name__ == '__main__':
    main()