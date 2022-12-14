import os
import torch
from torch import nn

from test import get_accuracy
from utils import get_data, log_string, data_split
from tqdm import tqdm
from model import MyModel, get_dataloader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_DIR = None
LR_DECAY_STEPS = 10


def _train_one_epoch(dataloader, net, optimizer, loss):
    for X, y in tqdm(dataloader):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat.view(-1), y)
        l.backward()
        optimizer.step()


def _train(train_dataloader, net: 'torch.nn.Module', optimizer, loss, scheduler, num_epochs, mean, std):
    for epoch in range(num_epochs):
        _train_one_epoch(train_dataloader, net, optimizer, loss)
        with torch.no_grad():
            get_accuracy(train_dataloader, net, loss, mean, std, LOG_DIR, epoch=epoch)
            scheduler.step()
            if not (epoch + 1) % LR_DECAY_STEPS:
                log_string('learning rate is %f now' % optimizer.state_dict()['param_groups'][0]['lr'], LOG_DIR)


def train(FLAGS):
    global LOG_DIR, LR_DECAY_STEPS
    LOG_DIR = FLAGS.log_dir
    num_epoch = FLAGS.max_epoch
    lr = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    gamma = FLAGS.lr_decay_rates
    LR_DECAY_STEPS = FLAGS.lr_decay_steps
    retrain_model = FLAGS.retrain_model
    mlps = FLAGS.mlps.copy()

    TRAIN_DATA_INDEX = 0
    # 数据处理
    df_train = get_data(TRAIN_DATA_INDEX)

    # 认为与price相关程度的绝对值小于 0.05 的不显著相关
    corr_data = df_train.corr()
    corr_low_columns = df_train.columns[corr_data.loc['price'].abs() < 0.05]
    df_train.drop(corr_low_columns, axis=1, inplace=True)

    df_train, df_test = data_split(df_train)
    df_y = df_train.pop('price')
    df_test_y = df_test.pop('price')
    price_mean = df_y.mean()
    price_std = df_y.std()
    df_y = (df_y - price_mean) / price_std
    df_test_y = (df_test_y - price_mean) / price_std

    train_dataloader = get_dataloader(batch_size, df_train, df_y)
    test_dataloader = get_dataloader(batch_size, df_test, df_test_y)

    in_feature = df_train.shape[1]
    mlps.insert(0, in_feature)
    net = MyModel(mlps).to(DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.01)
    # loss = nn.MSELoss()
    loss = nn.L1Loss()

    # 读检查点
    check_point_path = LOG_DIR + '/check_point.pt'
    if os.path.exists(check_point_path) and not retrain_model:
        load_dict = torch.load(check_point_path)
        net.load_state_dict(load_dict["net.state_dict()"])
        print("已存在检查点，故不进行模型训练。如果希望重新训练，请运行：python start.py --retrain_model=True")
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEPS, gamma=gamma)
        # 记录超参
        log_string(f"网络结构：{net}", LOG_DIR)
        log_string(f"初始学习率：{lr}", LOG_DIR)
        _train(train_dataloader, net, optimizer, loss, scheduler, num_epoch, price_mean, price_std)

        save_dict = {"net.state_dict()": net.state_dict(),
                     "price_mean": price_mean,
                     "price_std": price_std,
                     "corr_low_columns": corr_low_columns}
        torch.save(save_dict, check_point_path)

    # 获取测试集准确率
    get_accuracy(test_dataloader, net, loss, price_mean, price_std, LOG_DIR, is_train=False)
