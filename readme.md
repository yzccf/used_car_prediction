数据下载见[北航云盘](https://bhpan.buaa.edu.cn:443/link/3198637693396F7805FC058A54AC9A8A) ，并放在./car_data/ 文件夹中。

文件输出将放在./log/ 中

训练过程：在项目根目录
~~~bash
python train.py --retrain_model=True
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
```