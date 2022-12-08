数据下载见[北航云盘](https://bhpan.buaa.edu.cn:443/link/3198637693396F7805FC058A54AC9A8A) ，并放在./car_data/ 文件夹中。

文件输出将放在./log/ 中

训练过程：在项目根目录
~~~bash
python start.py --retrain_model=True --max_epoch=1000 --lr_decay_steps=100 --batch_size 512 --mlps 512 256 128 64 1
~~~

参数见:
```python
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
parser.add_argument('--mlps', type=int, default=[64, 32, 8, 1], nargs='+',
                    help='Linear layer structure in MyModel [default: 64 32 8 1]]')
```

过程记录：
1. 先是自己写的dataset和dataloader读取数据，发现不能收敛。暂未找到原因。
2. 由于自身笔记本的性能原因，用12w个训练数据中的1w个进行训练，得到MAE大概750。
3. 将mlps从64 32 8 1改成512 256 128 64 1之后，MAE从750（粗测）-> 650
4. 现在发现训练集上MAE大约为400，而测试集上650，判定发生了过拟合，因此加L2正则测试。
对optim加weight_decay=0.01，发现此时的训练和测试MAE均为750左右，模型预测能力下降。因此考虑**添加别的正则**。
根据[这个知乎](https://zhuanlan.zhihu.com/p/40814046)发现原因可能和adam优化器有关，因此换回SGD试试。
