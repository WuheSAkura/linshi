import os
import torch
import torch.nn as nn
#import model_utils
def getInput(args, data):
    input_list = [data['img']]
    if args.in_light: input_list.append(data['dirs'])
    if args.in_mask:  input_list.append(data['m'])
    return input_list

def parseData(args, sample, timer=None, split='train'):
    img, normal, mask = sample['img'], sample['normal'], sample['mask']
    ints = sample['ints']
    if args.in_light:
        dirs = sample['dirs'].expand_as(img)
    else: # predict lighting, prepare ground truth
        n, c, h, w = sample['dirs'].shape
        dirs_split = torch.split(sample['dirs'].view(n, c), 3, 1)
        dirs = torch.cat(dirs_split, 0)
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        img, normal, mask = img.cuda(), normal.cuda(), mask.cuda()
        dirs, ints = dirs.cuda(), ints.cuda()
        if timer: timer.updateTime('ToGPU')
    data = {'img': img, 'n': normal, 'm': mask, 'dirs': dirs, 'ints': ints}
    return data 

def getInputChanel(args):
    args.log.printWrite('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        args.log.printWrite('[Network Input] Adding Light direction as input')
        c_in += 3
    if args.in_mask:
        args.log.printWrite('[Network Input] Adding Mask as input')
        c_in += 1
    args.log.printWrite('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records} # 'args': args}
    torch.save(state,   os.path.join(save_path, 'checkp_{}.pth.tar'.format(epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_{}_rec.pth.tar'.format(epoch)))

def conv_ReLU(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.ReLU(inplace=True)
                )

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def outputConv(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True))

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )

def upconv(cin, cout):
    return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )


class ResidualBlock(nn.Module):
    """残差块 (替代原始连续卷积)"""

    def __init__(self, batchNorm, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = model_utils.conv(batchNorm, in_channels, out_channels, k=3, stride=stride, pad=1)
        self.conv2 = model_utils.conv(batchNorm, out_channels, out_channels, k=3, stride=1, pad=1, activation=None)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = model_utils.conv(batchNorm, in_channels, out_channels, k=1, stride=stride, pad=0,
                                             activation=None)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return nn.functional.leaky_relu(out + residual, 0.2)


class CBAM(nn.Module):
    """注意力模块 (加在特征输出前)"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels))
        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)
        channel_att = torch.sigmoid(self.fc(avg) + self.fc(max)).view(b, c, 1, 1)
        # Spatial
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * channel_att * spatial_att