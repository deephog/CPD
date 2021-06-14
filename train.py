import torch
import torch.nn.functional as fun
from torch.autograd import Variable
import sys
import numpy as np
import pdb, os, argparse
from datetime import datetime
import copy

from model.CPD_ResNet_models import CPD_ResNet
from model.CPD_models import CPD_VGG
from model.CPD_MobileNet_models import CPD_MobileNet
from data import get_loader
from utils import clip_gradient, adjust_lr

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=5000, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=str, default='mobile', help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

back_dir = '/home/hypevr/Desktop/data/projects/background/image/'
print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet == 'resnet':
    model = CPD_ResNet()
elif opt.is_ResNet == 'vgg':
    model = CPD_VGG()
elif opt.is_ResNet == 'mobile':
    model = CPD_MobileNet()

def DICE(inputs, targets, smooth=1):
    # comment out if your model contains a sigmoid or equivalent activation layer
    inputs = fun.sigmoid(inputs)
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice

model_dir = "./models/CPD_MobileNet_v3/"


##############################
checkpoint = None
load_pretrained = False
#############################


if checkpoint:
    checkpoint_dir = model_dir + 'CPD_' + str(checkpoint) + '.pth'
    model.load_state_dict(torch.load(checkpoint_dir))

if load_pretrained:
    model.load_state_dict(torch.load('CPD.pth'))

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

data_root = '/home/hypevr/data/projects/data/combined_human/'
image_root = data_root+'train/image/'
gt_root = data_root+'train/mask/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, fake_back_rate=0.1, back_dir=back_dir)
total_step = len(train_loader)

image_root_v = data_root+'val/image/'
gt_root_v = data_root+'val/mask/'
val_loader = get_loader(image_root_v, gt_root_v, batchsize=opt.batchsize, trainsize=opt.trainsize, fake_back_rate=0, back_dir=None)
total_step_v = len(val_loader)


CE = torch.nn.BCEWithLogitsLoss()



def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        atts, dets = model(images) #
        loss1 = CE(atts, gts)
        loss2 = CE(dets, gts)
        loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        print(str(i) + '/' + str(total_step))
        sys.stdout.write("\033[F")

        if i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Dice1: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss.data, ))


    save_path = 'models/' + opt.is_ResNet + '_random_0.1/'


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + 'CPD_%d.pth' % epoch)
    if (epoch+1) % 3 == 0:
        model.eval()
        for i, pack in enumerate(val_loader, start=1):
            images, gts = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()

            atts, dets = model(images)  #

            #dice1 = DICE(atts, gts)
            loss1 = CE(atts, gts)# + dice1
            # dice2 = DICE(dets, gts)
            loss2 = CE(dets, gts)# + dice2
            loss = loss1 + loss2

            print(str(i) + '/' + str(total_step_v))
            sys.stdout.write("\033[F")

            if i == total_step_v:
                print(
                    'validation phase: {} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss.data))

        torch.save(model.state_dict(), save_path + 'CPD_%d.pth' % epoch)

print("Let's go!")
for epoch in range(1, opt.epoch):
    if checkpoint:
        epoch += int(checkpoint)
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)

