'''
Function:
co-run training and infering
profiler recoding data when train and infer

use Pipe , after training ending,  cnn_train.py send message to profiller.py and cnn_infer.py to stop.

note: multiprocessing_exception.py is process cascade exception from children process to parent process
        https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent


inference latency prediction when co-run.

e.g. cnn config:
[{'arch': 'alexnet', 'workers': 1, 'batch_size': 1, 'image_size': 224, 'device': 'cuda'}, {'arch': 'mobilenet_v2', 'workers': 1, 'batch_size': 1, 'image_size': 224, 'device': 'cuda'}, {'arch': 'resnet152', 'workers': 1, 'batch_size': 1, 'image_size': 224, 'device': 'cuda'}]

'''

import json
import os
import multiprocessing as mp
import pandas as pd
import time
import multiprocessing_exception as mp_new  # procss.py is process cascade exception from children process to parent process
import traceback
import profiler
import sys
sys.path.append('../models')
sys.path.append('../datasets')
import cnn_infer, cnn_train, deeplab_v3, yolo_v5
import warnings
warnings.filterwarnings("ignore")

def date_time():
    s_l = time.localtime(time.time())
    dt = time.strftime("%Y%m%d", s_l)
    tm = time.strftime("%H%M%S", s_l)
    # print(date, tm )
    return dt, tm

def main():

    file = './config.json'
    dt, tm = date_time()
    log_file_name = 'profiler_log_' + dt + tm + '.csv'

    with open(file) as f:
        config = json.load(f)
    train_config_list, infer_config_list= config['train'],  config['infer']  #length: train: 36, infer: 12
    i=0
    profile_log = pd.DataFrame(columns = ['time_frame', 'train_configure', 'infer_configure','status', 'result'], index=None)

    profiling_num = 500
    ## start co-run train and infer.....
    for infer_config in infer_config_list:
        for train_config in train_config_list:

            print('======== round ', i, ': ==========')
            i += 1
            config, config_tr, config_inf = '', '', ''  #used as the configuration content and file name
            con_prf_a, con_prf_b = mp.Pipe()  # con_b in profiler message to main when complete task
            con_inf_a, con_inf_b = mp.Pipe()
            # con_prf_a.close()
            # con_inf_b.close()  # don't close it , otherwise pipe will break
            trn_queue = mp.Queue()
            inf_queue = mp.Queue()

            try:  # some models in the list main not available, so use it to bypass
                for key, value in train_config.items(): config_tr += (' ' + key + '_'+ str(value))
                for key, value in infer_config.items(): config_inf += (' ' + key + '_'+str(value))
                config = 'Train'+config_tr +' + '+ 'Infer'+ config_inf
                print(config)
                p1 = mp.Process(target= profiler.profile, args = (config, profiling_num, (con_prf_a,con_prf_b),))

                if train_config['arch'] == 'deeplab_v3':
                    p2 = mp_new.Process(target=deeplab_v3.work_train, args=(train_config,),
                                        kwargs=(dict(queue=trn_queue)))
                else:
                    p2 = mp_new.Process(target= cnn_train.work, args= (train_config,), kwargs=(dict(queue=trn_queue)))

                if infer_config['arch'] == 'yolo_v5s':
                    p3 = mp_new.Process(target = yolo_v5.work_infer, args = (infer_config, (con_inf_a,con_inf_b),), kwargs=(dict(queue=inf_queue)))
                else:
                    p3 = mp_new.Process(target = cnn_infer.work, args = (infer_config, (con_inf_a,con_inf_b),), kwargs=(dict(queue=inf_queue)))

                p_list=[p1, p2, p3]
                infer_info = 'na'

                for p in p_list: p.start()

                while p2.is_alive() or p3.is_alive():

                    # below to break when profiler complete job
                    if con_prf_b.poll():
                        msg= con_prf_b.recv()
                        con_prf_b.close()
                        if msg == 'done':
                            print('Main: get "done" notice from profiler! ')
                            con_inf_a.send('stop')  # notice infer to stop
                            con_inf_a.close()

                            while not inf_queue.empty():  # take the last queue data (latency)
                                inf_queue.get()  # take the rest until last one
                            infer_info = inf_queue.get()
                            print('Main: Get data from infer:', infer_info)

                        print('Main: Terminate training and inference', '\n'*5)
                        profile_log.to_csv(os.path.join('../result/log', log_file_name), index_label=None, mode='w')
                        print('log file saved in:', os.path.join('../result/log', log_file_name))
                        p2.terminate()
                        p3.terminate()
                        break

                    ######## below codes for receiving exception from subprocesser #########
                    if p2.exception:
                        print('Terminate all processes due to train process exception. ')
                        error, trc_back = p2.exception
                        p3.terminate()
                        p2.terminate()
                        p1.terminate()
                        raise ChildProcessError(trc_back)
                        profile_log.to_csv(os.path.join('../result/log', log_file_name), index_label=None, mode='w')
                        print('log file saved in:', os.path.join('../result/log', log_file_name))
                        break
                    if p3.exception:
                        print('Terminate all processes, due to infer process exception.')
                        error, trc_back = p3.exception
                        p3.terminate()
                        p2.terminate()
                        p1.terminate()
                        raise ChildProcessError(trc_back)
                        profile_log.to_csv(os.path.join('../result/log', log_file_name), index_label=None, mode='w')
                        print('log file saved in:', os.path.join('../result/log', log_file_name))
                        break

                for p in p_list:
                    p.join()

                for p in p_list:
                    p.terminate()

                profile_log.loc[i]= [time.time(), train_config, infer_config, 'Sucess',infer_info]  # add one row of data
                profile_log.to_csv(os.path.join('../result/log', log_file_name), index_label=None, mode='w')

            except Exception:
                print('traceback:', traceback.format_exc())
                profile_log.loc[i] = [time.time(), train_config, infer_config, 'Fail', trc_back]
                profile_log.to_csv(os.path.join('../result/log', log_file_name), index_label=None, mode='w')

if __name__ == '__main__':
    main()