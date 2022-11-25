import torch

from utils import log_string

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_accuracy(dataloader, net, loss, mean, std, log_dir, epoch=0, is_train=True):
    """
    输入训练好的net，测试对输入数据集的准确率
    """
    total_loss = 0
    total_n = 0
    flag = True
    truth, predict = 0, 0
    for X, y in dataloader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        y_hat = net(X).view(-1)
        l = loss(y_hat * std + mean, y * std + mean)
        if flag:
            flag = False
            truth, predict = y[0], y_hat[0]
        total_loss += l * X.shape[0]
        total_n += X.shape[0]
        if is_train:
            # 训练集就随便找一个batch简单算一下就行
            break
    if is_train:
        log_string(f'epoch {epoch}\'s {loss.__class__.__name__} on train dataset'
                   f' is {total_loss / total_n}, for example: y is {truth} and prediction is {predict}', log_dir)
    else:
        log_string(f'Finally, the {loss.__class__.__name__} on test dataset is {total_loss / total_n},'
                   f' for example: y is {truth} and prediction is {predict}', log_dir)
