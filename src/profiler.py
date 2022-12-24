import json
import os
import re
import subprocess
import pandas as pd
import time


def cpu_freq():
    dev=json.load(open(r'./device.json','r'))
    cpu_freq_dict ={}
    for i in range(os.cpu_count()):

        cpu_name = 'cpu'+str(i)
        arg = dev['cpu'][cpu_name]['freq_now']
        cpu_freq_dict[cpu_name+'_freq']= subprocess.check_output(['cat', arg]).decode('utf-8').strip()
    return cpu_freq_dict


def cpu_usage():
    p = subprocess.Popen(["top", "n1", "b"], \
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    threads_info = stdout.decode('utf-8').split('\n')
    cpu_usg=threads_info[2].split(',')
    cpu_usg_val= re.findall(r'[0-9]+[.][0-9]+', str(cpu_usg))  # parse value
    cpu_usg_col = re.findall('[a-zA-Z]+', str(cpu_usg))[2:] # [:-2]: take off string of '%Cpu(s)'
#     cpu_usg = [float(value.strip("%"))/float(cpu_usg[0].strip("%"))*100 for value in cpu_usg]
#     thrd_num= len(threads_info) -5
#     cpu_usg_col.append('thrd_num')
#     cpu_usg_val.append(thrd_num)
    cpu_usg_dict = dict(zip(cpu_usg_col, cpu_usg_val))

    return cpu_usg_dict

def date_time():
    s_l = time.localtime(time.time())
    dt = time.strftime("%Y%m%d", s_l)
    tm = time.strftime("%H%M%S", s_l)
    # print(date, tm )
    return dt, tm

def fmt_to_stmp(tm,*arg):
    fmt= "%Y-%m-%d %H:%M:%S" if not arg else arg[0]
    dt, ms =tm.split('.')  # to process with milli-second
    timeArray =time.strptime(dt, fmt)
    timeStamp = time.mktime(timeArray)+int(ms)/1000
    return timeStamp

def grab_gpu_data(arg):
    query_dic = {'time_stamp':'timestamp', 'gpu_name':'name', 'index':'index','gpu_power':'power.draw',\
                 'gpu_freq':'clocks.gr', 'gpu_mem_freq':'clocks.mem','gpu_temp':'temperature.gpu',\
                 'gpu_util%':'utilization.gpu', 'gpu_mem_util%':'utilization.memory',\
                 'gpu_mem_total':'memory.total', 'gpu_mem_used':'memory.used'}

    pos, i = -1, 0 # the position of time_stamp for converting
    query = ''  #create an empty string
    for item in arg:
        if item == 'time_stamp':
            pos = i
        i+=1
        query+=query_dic[item]+','

    query='--query-gpu='+query[:-1]  # add query string's head and cut tail, which tail is ','
    nvidia_smi = "nvidia-smi"
    p = subprocess.Popen([nvidia_smi, query, "--format=csv,noheader,nounits"], stdout=subprocess.PIPE) # close_fds= True
    stdout,  stderror = p.communicate()
    output = stdout.decode('UTF-8').strip()
    output = output.split(',')  #split the returned string and convert to list
    output[pos]= fmt_to_stmp(output[pos], "%Y/%m/%d %H:%M:%S") # convert date-time to stamp format, format in "%Y/%m/%d %H:%M:%S"
    return dict(zip(arg, output))

def save_log(data,config):
    dt, tm = date_time()
    log_dir = '../result/log/' + dt
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    log_file = config + '_'+ dt + tm
    data.to_csv(os.path.join(log_dir,log_file)+ '.csv')
    # data.to_json(os.path.join(log_dir, log_file) + '.json')(orient='index') #default: column

def record(config, pipe):
    con_a, con_b = pipe
    con_b.close()  ##  only receive message from train, 关闭出去端
    gpu_col = ['time_stamp', 'gpu_power', 'gpu_freq', 'gpu_mem_freq', 'gpu_temp', 'gpu_util%', 'gpu_mem_util%', 'gpu_name']
    gpu_dict = grab_gpu_data(gpu_col)
    cpu_usg_dict = cpu_usage()
    cpu_freq_dict = cpu_freq()
    # print(cpu_usg_dict, cpu_freq_dict, gpu_dict)

    cpu_col = ['timestamp'] + list(cpu_usage().keys()) + list(cpu_freq().keys())  # create the cpu, gpu's columns of dataframe
    cpu = pd.DataFrame(columns=cpu_col, index=None)
    gpu = pd.DataFrame(columns=gpu_col, index=None)

    col_select = ['gpu_power', 'gpu_freq', 'gpu_mem_freq', 'gpu_temp', 'gpu_util%', 'gpu_mem_util%', 'us', 'sy',
           'cpu0_freq', 'cpu5_freq']

    start = time.time()
    i = 0

    while True:
        try:
            # cpu data collection
            cpu_usg_dict = cpu_usage()
            cpu_freq_dict = cpu_freq()
            val = [time.time()] + list(cpu_usg_dict.values()) + list(cpu_freq_dict.values())
            cpu.loc[i, cpu_col] = val

            # gpu data collection
            val = list(grab_gpu_data(gpu_col).values())
            gpu.loc[i, gpu_col] = val

            i += 1
            time.sleep(0.418)

            print('\r', i, ' instances.', end='')
        except:  # press stop to break
            print()
            print('Profile collecting done!')
            data = gpu.join(cpu)
            save_log(data, config)
            # print(cpu_usg_dict)
            break


        if time.time() - start >= 172800:  # >48 hours, then end. time.time() is in second
            data = gpu.join(cpu)
            save_log(data, config)
            print('Profile collecting stop due to time over 48 hours')
            print()
            break  # record 10 sec.

        try:  # if train end, then profiling end
            if con_a.poll():  # Roy: use poll to solve con_a.recr blocking issue
                msg = con_a.recv()
                if msg =='end':  # signal from training process to end, ROy added on 12092022
                    data = gpu.join(cpu)
                    # save_log(data, config)
                    print()
                    print('Profile collecting done!')
                    con_a.close()
                break

        # except EOFError as e:
        except Exception as e:
            print(e)
            break

    print('Time elapsed: ', time.time() - start, 'sec.')
    print()

if __name__ == '__main__':
    record('na')