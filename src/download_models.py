from torchstat import stat
from torchsummary import summary
from torchvision import models
import torch

cnn_model_list = ['inception_v3']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101','deeplabv3_mobilenet_v3_large']
model_list = cnn_model_list + deeplab_model_list + yolo_model_list

for model_name in model_list:

    # if model_name in deeplab_model_list:
    #     print(model_name)
    #     model_func = 'models.segmentation.'+model_name
    #     model = eval(model_func)(pretrained=True)
    #     stat(model,(3,224,448))
    #     print('\n' * 5)

    if model_name in cnn_model_list:
        print(model_name)
        model_func = 'models.' + model_name
        model = eval(model_func)(pretrained=True)
        stat(model,(3,448,448))
        print('\n'*5)

    # if model_name in yolo_model_list:
    #     model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device =0)
    #     summary(model,(3,224,224))
    #     print('\n'*5)