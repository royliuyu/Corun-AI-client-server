'''
Function:
co-run training and infering
profiler recoding data when train and infer

use Pipe , after training ending,  cnn_train.py send message to profiller.py and cnn_infer.py to stop.

note: multiprocessing_exception.py is process cascade exception from children process to parent process
        https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent


inference latency prediction when co-run.

'''

import json

import multiprocessing as mp
import pandas as pd
import time
import multiprocessing_exception as mp_new  # procss.py is process cascade exception from children process to parent process
import traceback
import profiler


def main():

    file = './config.json'
    with open(file) as f:
        config = json.load(f)
    train_config_list, infer_config_list= config['train'],  config['infer']  #length: train: 36, infer: 12
    i=0
    profile_log = pd.DataFrame(columns = ['time_frame', 'train_configure', 'infer_configure','status', 'result'], index=None)

    ## start co-run train and infer.....
    for infer_config in infer_config_list:
        for train_config in train_config_list:

            print('======== round ', i, ': ==========')
            i += 1
            config, config_tr, config_inf = '', '', ''  #used as the configuration content and file name
            con_a, con_b = mp.Pipe()  # con_b in tain send message to con_a in profiler
            con_1, con_2 = mp.Pipe()  # for control infer process, train process inform infer proess stop after training finish
            trn_queue = mp.Queue()
            inf_queue = mp.Queue()

            try:  # some models in the list main not available, so use it to bypass
                for key, value in train_config.items(): config_tr += (' ' + key + '_'+ str(value))
                for key, value in infer_config.items(): config_inf += (' ' + key + '_'+str(value))
                config = 'Train'+config_tr +' + '+ 'Infer'+ config_inf
                print(config)
                p1 = mp.Process(target= profiler.record, args = (config, (con_a,con_b),))
                p2 = mp_new.Process(target= train.work, args= (train_config,(con_a,con_b,con_1, con_2),), kwargs=(dict(queue=trn_queue)))
                p3 = mp_new.Process(target = infer.work, args = (infer_config,(con_1, con_2),), kwargs=(dict(queue=inf_queue)))
                p_list=[p1, p2, p3]
                infer_info = 'na'

                for p in p_list: p.start()

                ######## below codes for receiving exception from subprocesser #########
                while p2.is_alive() or p3.is_alive():
                    if not inf_queue.empty():
                        # print('Infer queue has data. kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
                        infer_info = inf_queue.get()

                    if p2.exception:
                        print('Terminate all processes due to train process exception. ')
                        error, trc_back = p2.exception
                        p3.terminate()
                        p2.terminate()
                        p1.terminate()
                        raise ChildProcessError(trc_back)
                        profile_log.to_csv('../result/log/profiler_log.csv', index_label=None, mode='w')
                        break
                    if p3.exception:
                        print('Terminate all processes, due to infer process exception.')
                        error, trc_back = p3.exception
                        p3.terminate()
                        p2.terminate()
                        p1.terminate()
                        raise ChildProcessError(trc_back)
                        profile_log.to_csv('../result/log/profiler_log.csv', index_label=None, mode='w')
                        break

                for p in p_list:
                    p.join()

                for p in p_list:
                    p.terminate()

                profile_log.loc[i]= [time.time(), train_config, infer_config, 'Sucess',infer_info]  # add one row of data
                profile_log.to_csv('../result/log/profiler_log.csv',index_label=None, mode='w')

            except Exception:
                print('traceback:', traceback.format_exc())
                profile_log.loc[i] = [time.time(), train_config, infer_config, 'Fail', trc_back]
                profile_log.to_csv('../result/log/profiler_log.csv',index_label=None, mode='w')

if __name__ == '__main__':
    main()