import re
import pandas as pd
import numpy as np

file = '../result/log/profiler_log_20230211000401.csv'

df = pd.read_csv(file)
train_models = np.unique(df['train_configure'])
for model in train_models:
    data = df.where(df['train_configure']== model).dropna()
    result_dict =(data.loc[:,('infer_configure','result')])
#     print(result_dict)
#     print(len(result_dict))
    for a in result_dict:
        print(a)
        print(a)
#     train_dur = re.findall(r'\'duration_sec\':(.+)',result_dict)
#     print(result_dict.iloc[1])