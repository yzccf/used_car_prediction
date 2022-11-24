import pandas as pd
import numpy as np
import os
import time

corr_low_columns = None


def data_process(df_data, is_train=True):
    # 销售编号和交易名称逻辑上不包含信息，所以删除
    df_data.drop(['SaleID', 'name'], axis=1, inplace=True)
    # offerType只有一个值，不包含信息；seller数据太偏，信息不准所以删除
    df_data.drop(['offerType', 'seller'], axis=1, inplace=True)

    # 将日期转成时间戳的天数
    df_data['creatDate'] = (pd.to_datetime(df_data['creatDate'], format='%Y%m%d', errors='coerce') - np.datetime64(
        '1970-01-01T00:00:00Z')) // np.timedelta64(1, 'D')
    df_data['regDate'] = (pd.to_datetime(df_data['regDate'], format='%Y%m%d', errors='coerce') - np.datetime64(
        '1970-01-01T00:00:00Z')) // np.timedelta64(1, 'D')
    df_data = df_data.fillna(df_data.mode().iloc[0, :])
    # 替换掉空值之后取得从车注册到卖出的时间
    df_data['used_time'] = df_data['creatDate'] - df_data['regDate']

    # 此外观察到creatDate相关度基本为0，因此regDate和used_time保留一个即可
    df_data.drop(['regDate'], axis=1, inplace=True)

    df_data['used_time'] = df_data['used_time'].astype('int32')
    df_data = pd.get_dummies(df_data, columns=['gearbox', 'notRepairedDamage'])
    # 将std较大的column归一化
    df_data[['power', 'used_time']] \
        = (df_data[['power', 'used_time']] - df_data[['power', 'used_time']].mean()) / df_data[
        ['power', 'used_time']].std()
    return df_data


FOUT = None


def log_string(out_str, log_dir):
    global FOUT
    if not log_dir:
        print(out_str)
        return
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not FOUT:
        FOUT = open(os.path.join(log_dir + 'train' + time.strftime('%m-%d_%H:%M:%S', time.localtime()) + '.txt'), 'w')
    FOUT.write(out_str + '\n')
    FOUT.flush()


file_path = ['./car_data/train.csv', './car_data/test.csv', './car_data/submission.csv']
na_values = ['-', 'na']


def get_data(type: int):
    df_data = pd.read_csv(file_path[type], sep=' ', na_values=na_values)
    if type == 0:
        return data_process(df_data)
    else:
        return data_process(df_data, False)
