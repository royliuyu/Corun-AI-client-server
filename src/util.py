import time
import os
import csv

def date_time():
    s_l = time.localtime(time.time())
    dt = time.strftime("%Y%m%d", s_l)
    tm = time.strftime("%H%M%S", s_l)
    # print(date, tm )
    return dt, tm

def logger_by_date(data_in_row, dir, file_name_prefix):  # save in csv format
    data, _ = date_time()
    file_path = os.path.join(dir,file_name_prefix+ data+ '.csv')
    if os.path.exists(file_path):
        with open(file_path, 'a', newline='') as csvf:
            writer =  csv.writer(csvf)
            writer.writerow(data_in_row)
    else:
        with open(file_path, 'w', newline='') as csvf:
            writer =  csv.writer(csvf)
            writer.writerow(data_in_row)


def fmt_to_stmp(tm,*arg):
    fmt= "%Y-%m-%d %H:%M:%S" if not arg else arg[0]
    dt, ms =tm.split('.')  # to process with milli-second
    timeArray =time.strptime(dt, fmt)
    timeStamp = time.mktime(timeArray)+int(ms)/1000
    return timeStamp

def save_log(data,file_name):
    dt, tm = date_time()
    log_dir = '../result/log/' + dt
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    log_file = file_name + '_'+ dt + tm
    data.to_csv(os.path.join(log_dir,log_file)+ '.csv')