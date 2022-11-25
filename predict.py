import os

import torch
from tqdm import tqdm
import csv

from model import get_dataloader, MyModel
from utils import get_data

out_csv = 'submission.csv'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_DIR = None
TEST_DATA_INDEX = 1


def _predict(dataloader, net, log_dir, price_mean, price_std, batch_size):
    net.eval()
    with open(log_dir + out_csv, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["SaleID", "price"])
        i = 0
        for X in tqdm(dataloader):
            X = X[0].to(DEVICE)
            y = net(X).view(-1)
            y = y.detach().cpu()
            prediction = torch.where((y * price_std + price_mean) < 0.0, torch.tensor(price_mean),
                                     y * price_std + price_mean)
            prediction = prediction.numpy().tolist()
            writer.writerows(list(zip([_ for _ in range(i, i + batch_size)], prediction)))
            i += batch_size


def predict(FLAGS):
    global LOG_DIR
    LOG_DIR = FLAGS.log_dir
    batch_size = FLAGS.batch_size

    df_pred = get_data(TEST_DATA_INDEX)

    check_point_path = LOG_DIR + '/check_point.pt'
    assert os.path.exists(check_point_path), "未找到模型参数，可能是未执行过train就直接predict"

    load_dict = torch.load(check_point_path)
    corr_low_columns = load_dict["corr_low_columns"]
    df_pred.drop(corr_low_columns, axis=1, inplace=True)
    test_dataloader = get_dataloader(batch_size, df_pred, is_train=False)

    in_feature = df_pred.shape[1]
    net = MyModel([in_feature, 64, 32, 8, 1]).to(DEVICE)

    net.load_state_dict(load_dict["net.state_dict()"])
    price_mean = load_dict["price_mean"]
    price_std = load_dict["price_std"]

    _predict(test_dataloader, net, LOG_DIR, price_mean, price_std, batch_size)
    print("-" * 10)
    print(f"预测文件已经输出到 {LOG_DIR} 中")
