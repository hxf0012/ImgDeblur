import torch
import torch.nn as nn
from .common import *
# 全连接层 输入200  隐藏层 1000 输出 1
def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):

# 输入 隐藏
    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden,bias=True))
    model.add(nn.ReLU6())
# 隐藏 输出
    model.add(nn.Linear(num_hidden, num_output_channels))
    model.add(nn.Softmax())
#
    return model











