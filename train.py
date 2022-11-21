import os
import torch
from torch import nn
import argparse
from utils import get_data, log_string
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import MyModel, get_dataloader
from predict import predict

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='./log/', help='Dump dir to save model checkpoint or log '
                                                        '[example: ./log/]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.001]')
parser.add_argument('--lr_decay_rates', type=float, default=0.8, help='Decay rates for lr decay [default: 0.8]')
parser.add_argument('--lr_decay_steps', type=int, default=10,
                    help='When to decay the learning rate (in epochs) [default: 10]')
parser.add_argument('--retrain_model', type=bool, default=False,
                    help='Either to read the checkpoint or not [default: False]')

FLAGS = parser.parse_args()

log_dir = FLAGS.log_dir
num_epoch = FLAGS.max_epoch
lr = FLAGS.learning_rate
batch_size = FLAGS.batch_size
gamma = FLAGS.lr_decay_rates
lr_decay_steps = FLAGS.lr_decay_steps
retrain_model = FLAGS.retrain_model

TRAIN_DATA = 0
# 数据处理
df_train = get_data(TRAIN_DATA)

df_y = df_train.pop('price')
price_mean = df_y.mean()
price_std = df_y.std()
df_y = (df_y - price_mean) / price_std

in_feature = df_train.values.shape[1]

my_dataloader = get_dataloader(batch_size, df_train[:10000], df_y[:10000])

# df_train 内存释放
df_train = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MyModel([in_feature, 64, 32, 8, 1]).to(device)


def train_one_epoch(dataloader, net, optimizer, loss):
    for X, y in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat.view(-1), y)
        l.backward()
        optimizer.step()


def train(dataloader: 'torch.utils.data.DataLoader', net: 'torch.nn.Module', optimizer, loss, scheduler, num_epochs):
    for epoch in range(num_epochs):
        train_one_epoch(dataloader, net, optimizer, loss)
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                l = loss(y_hat.view(-1), y)
                log_string(f'epoch{epoch}\'s MSE loss is {l}, for example: y is {y[0]} and prediction is {y_hat[0][0]}', log_dir)
                scheduler.step()
                if not (epoch + 1) % lr_decay_steps:
                    log_string('learning rate is %f now' % optimizer.state_dict()['param_groups'][0]['lr'], log_dir)
                break


# 读检查点
check_point_path = log_dir + '/check_point.pt'
if os.path.exists(check_point_path) and not retrain_model:
    net.load_state_dict(torch.load(check_point_path))
else:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=gamma)

    # 记录超参
    log_string(f"网络结构：{net}", log_dir)
    log_string(f"初始学习率：{lr}", log_dir)

    train(my_dataloader, net, optimizer, loss, scheduler, num_epoch)

if not os.path.exists(check_point_path):
    torch.save(net.state_dict(), check_point_path)

TEST_DATA = 1
df_test = get_data(TEST_DATA)
test_dataloader = get_dataloader(batch_size, df_test)

predict(test_dataloader, net, log_dir, price_mean, price_std, batch_size)
print("-" * 10)
print(f"预测文件已经输出到 {log_dir} 中")
