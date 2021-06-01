# -*- coding：utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os

from networks.fcn import fcn
import glob

from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM


# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[896, 896], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[27, 27], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="datasets/", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results')
opt = parser.parse_args()
# print(opt)

# 环境
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
warnings.filterwarnings("ignore")

# 文件打开和保存路径
files_source = glob.glob(os.path.join(opt.data_path, '*.jpg')) # 此处更改图像类型png或者jpg
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)


'''新的网络结构 retinet'''
class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d(int((kernel_size - 1) / 2))
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.reflection(x)
        out = self.conv(out)
        # out = self.relu(out)

        return out


class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        self.A1 = ConvBlock(1, 64, kernel_size=11)
        self.B1 = ConvBlock(64, 64, kernel_size=3)
        self.B2 = ConvBlock(64, 64, kernel_size=3)
        self.B3 = ConvBlock(64, 64, kernel_size=3)
        self.C1 = ConvBlock(64, 1, kernel_size=5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        A1 = self.A1(x)
        A1 = self.relu(A1)

        B1 = self.B1(A1)
        B1 = self.relu(B1)
        B2 = self.B2(B1)
        B2 = self.relu(B2)
        B3 = self.B3(B2)
        B3 = self.relu(B3)

        out = self.C1(B3)

        return out

# start #image
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'  
    LR = 0.01  
    num_iter = opt.num_iter  
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]


    _, imgs = get_image(path_to_image, -1)  # load image and convert to np.
    y = np_to_torch(imgs).type(dtype)


    img_size = imgs.shape  
    print("or_img_size",img_size)

    padh, padw = opt.kernel_size[0] - 1, opt.kernel_size[1] - 1
    opt.img_size[0], opt.img_size[1] = img_size[1] + padh, img_size[2] + padw  # 优化图像尺寸等于原尺寸加上模糊核大小减一


    '''
    x_net: 图片输入
    '''
    net_input = y.type(dtype).detach()
    print("net_input", net_input.shape)
    net = DeblurNet()
    net = net.type(dtype)

    '''
    k_net: 模糊核输入
    '''
    n_k = 200  # 模糊核维数
    # 输入
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype).detach()  # 1D噪声 zk 200维
    net_input_kernel = net_input_kernel.squeeze()  # net_input_kernel.squeeze_()
    # 网络
    net_kernel = fcn(n_k, opt.kernel_size[0] * opt.kernel_size[1])  # 和模糊核一样大小的zk
    net_kernel = net_kernel.type(dtype)

    # Losses 损失函数
    mse = torch.nn.MSELoss().type(dtype)

    ssim = SSIM().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': net_kernel.parameters(), 'lr': 1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

   
    for step in tqdm(range(num_iter)):

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()
       
        # get the network output
        out_x = net(net_input)

        out_x_np = torch_to_np(out_x)
        out_x_np = out_x_np.squeeze()
        out_x_np /= np.max(out_x_np)

        padimg = np.pad(out_x_np, ( (13, 13), (13, 13)), 'constant', constant_values=0)
        padimg = np_to_torch(padimg).type(dtype)

        padimg_m = padimg.view(-1, 1, opt.img_size[0], opt.img_size[1])

        out_k = net_kernel(net_input_kernel)
        out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])

        #模糊核与清晰图卷积
        out_y = nn.functional.conv2d(padimg_m, out_k_m, padding=0, bias=None)


        if step < 1000:
            total_loss = mse(out_y, y) 

        else:
            total_loss = 1 - ssim(out_y, y)

        
        total_loss.backward()
        optimizer.step()

 
        if (step + 1) % opt.save_frequency == 0: 
            save_path = os.path.join(opt.save_path, '%s_x.png' % imgname)
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            imsave(save_path, out_x_np)

            save_path = os.path.join(opt.save_path, '%s_k.png' % imgname)
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            imsave(save_path, out_k_np)

            torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))
            torch.save(net_kernel, os.path.join(opt.save_path, "%s_knet.pth" % imgname))
