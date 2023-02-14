# import re
# def str2dict(file_name):  # file name with .csv
#     '''
#     e.g. input: 'Train arch_vgg16 workers_1 epochs_5 batch_size_64 image_size_224 device_cuda + Infer arch_alexnet workers_1 batch_size_1 image_size_224 device_cuda_20230211082435.csv'
#     output:
#     {'arch': 'vgg16', 'workers': '1', 'epochs': '5', 'batch_size': '64', 'image_size': '224', 'device': 'cuda'} {'arch': 'alexnet', 'workers': '1', 'batch_size': '1', 'image_size': '224', 'device': 'cuda'}
#
#     '''
#     prefix, _ = re.findall(r'(.+)(_[0-9]+).csv', file_name)[0]
#     arg_list = ['arch', 'workers', 'epochs', 'batch_size', 'image_size', 'device']
#     config = prefix.split('+')
#     train_args = re.findall(r'Train (.+)', config[0].strip())[0].split(' ')
#     infer_args = re.findall(r'Infer (.+)', config[1].strip())[0].split(' ')
#     train_dict = {}
#     infer_dict = {}
#     for arg in arg_list:
#         for arg_val in train_args:
#             if re.match(arg, arg_val):
#                 val = re.findall(arg + '_(.+)', arg_val)[0]
#                 train_dict[arg] = val
#         for arg_val in infer_args:
#             if re.match(arg, arg_val):
#                 val = re.findall(arg + '_(.+)', arg_val)[0]
#                 infer_dict[arg] = val
#     return train_dict, infer_dict
#
#
# def dict_str2value(dict_old):
#     model_list = ['deeplab_v3', 'yolo_v5s', 'alexnet', 'densenet201', 'mobilenet_v2', 'resnet152', 'shufflenet_v2_x1_0',
#                   'squeezenet1_0', 'vgg16']
#     model_map = {}
#     for i, model_name in enumerate(model_list): model_map[model_name] = i
#     dict_new = {}
#     for arg, val in dict_old.items():
#         if arg == 'arch':
#             dict_new['arch'] = model_map[val]
#         elif arg == 'device':
#             dict_new['device'] = 1 if dict_old['device'] == 'cuda' else 0
#         else:
#             dict_new[arg] = int(val)
#     return dict_new
#
#
# def dict_str2value_partial(dict_old):
#     model_list = ['deeplab_v3', 'yolo_v5s', 'alexnet', 'densenet201', 'mobilenet_v2', 'resnet152', 'shufflenet_v2_x1_0',
#                   'squeezenet1_0', 'vgg16']
#     model_map = {}
#     for i, model_name in enumerate(model_list): model_map[model_name] = i
#     dict_new = {}
#     for arg, val in dict_old.items():
#         if arg == 'arch':
#             dict_new['arch'] = val
#         elif arg == 'device':
#             dict_new['device'] = val
#         else:
#             dict_new[arg] = int(val)
#             dict_new[arg] = int(val)
#     return dict_new
#
#
# def filename2config(file_name):
#     '''
#     'e.g. input  (file_name prefix):
#          'Train arch_vgg16 workers_1 epochs_3 batch_size_32 image_size_224 device_cuda + Infer arch_resnet50 workers_1 epochs_3 batch_size_1 image_size_224 device_cuda'
#     output:
#         'Train: arch_vgg16, workers_1, epochs_3, batch_size_32, image_size_224, device_cuda \r
#         'Infer, arch_resnet50, workers_1, epochs_3, batch_size_1, image_size_224, device_cuda'
#     '''
#     tasks = file_name.split('+')  # saperate to two string of "Train" and "Infer"
#     task_names = ''
#
#     for i, task in enumerate(tasks):
#
#         config_list = []  # task: string of train/infer, config[i]: list of batch_size, arch , etc
#         config_list = task.strip().split(' ')
#         config_str = config_list[0] + ' '  # add 'Train:', or 'infer'
#         for i in range(1, len(config_list)):  # skip first element string of "Train"
#             config_str += config_list[i] + ', '
#         config_str = config_str[:-2]
#         task_names = task_names + config_str + '+'
#     return task_names[:-1]
