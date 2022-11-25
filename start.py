import argparse

from predict import predict
from train import train

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
parser.add_argument('--mlps', type=int, default=[64, 32, 8, 1], nargs='+',
                    help='Linear layer structure in MyModel [default: 64 32 8 1]]')


FLAGS = parser.parse_args()

train(FLAGS)
predict(FLAGS)
