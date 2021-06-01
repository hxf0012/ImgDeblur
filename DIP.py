# -*- coding：utf-8 -*-
from __future__ import print_function
import argparse
import os
from networks.skip import skip
from networks.fcn import fcn
import glob
from skimage.io import imsave
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM

# 设置参数----这些参数都是可调的
parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=1000, help='number of epochs of training') # 训练次数
parser.add_argument('--img_size', type=int, default=[896, 896], help='size of each image dimension') # 图片大小
parser.add_argument('--kernel_size', type=int, default=[17, 17], help='size of blur kernel [height, width]') # 卷积核大小
parser.add_argument('--data_path', type=str, default="datasets/", help='path to blurry image') # 数据路径
parser.add_argument('--save_path1', type=str, default="results/img/", help='path to save results') # 生成的图片、卷积核以及模型保存路径
parser.add_argument('--save_path2', type=str, default="results/kernal/", help='path to save results')
parser.add_argument('--save_path3', type=str, default="results/other/", help='path to save results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results') # 多少轮保存一次
opt = parser.parse_args()
#print(opt)


# # 环境
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor
warnings.filterwarnings("ignore")

# 文件打开和保存路径
files_source = glob.glob(os.path.join(opt.data_path, '*.jpg'))    # 记得修改文件类型
files_source.sort()
save_path1= opt.save_path1
os.makedirs(save_path1, exist_ok=True)
save_path2= opt.save_path2
os.makedirs(save_path2, exist_ok=True)
save_path3= opt.save_path3
os.makedirs(save_path3, exist_ok=True)


# start 
for f in files_source:
    INPUT = 'noise'
    pad = 'reflection'   #镜像填充
    LR = 0.01  
    num_iter = opt.num_iter 
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]


    _, imgs = get_image(path_to_image, -1) # load image and convert to np.
    y = np_to_torch(imgs).type(dtype)

    img_size = imgs.shape  #  shape读取图像尺寸

    padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw      # 优化图像尺寸等于原尺寸加上模糊核大小减一

    '''
    x_net: 图片输入
    '''
    input_depth = 8
    # 随机噪声 zx 输入， 8路，大小L+M-1
    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype).detach()
    # 调用Skip网络
    net = skip( input_depth, 1,
                num_channels_down = [128, 128, 128, 128, 128],
                num_channels_up   = [128, 128, 128, 128, 128],
                num_channels_skip = [16, 16, 16, 16, 16],
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

    '''
    k_net: 模糊核输入
    '''
    n_k = 200  # 模糊核维数
    # 输入
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype).detach()   # 1D噪声 zk 200维
    net_input_kernel = net_input_kernel.squeeze() # net_input_kernel.squeeze_()
    # 网络
    net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])  # 和模糊核一样大小的zk
    net_kernel = net_kernel.type(dtype)

    # Losses 损失函数
    mse = torch.nn.MSELoss().type(dtype)


    ssim = SSIM().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    # start 
    total_loss1 = []
    for step in tqdm(range(num_iter)):
        net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)

        out_k = net_kernel(net_input_kernel)
        out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])

        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

        if step < 100:
            total_loss = mse(out_y,y)
        else:
            total_loss = 1-ssim(out_y, y)

        total_loss.backward()
        optimizer.step()


        # 保存结果
        if (step+1) % opt.save_frequency == 0:    

            save_path1 = os.path.join(opt.save_path1, '%s_x.png'%imgname)
            out_x_np = torch_to_np(out_x)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np[padh//2:padh//2+img_size[1], padw//2:padw//2+img_size[2]]
            imsave(save_path1, out_x_np)

            save_path2 = os.path.join(opt.save_path2, '%s_k.png'%imgname)
            out_k_np = torch_to_np(out_k_m)
            out_k_np = out_k_np.squeeze()
            out_k_np /= np.max(out_k_np)
            imsave(save_path2, out_k_np)

            torch.save(net, os.path.join(opt.save_path3, "%s_xnet.pth" % imgname))
            torch.save(net_kernel, os.path.join(opt.save_path3, "%s_knet.pth" % imgname))
