import torch
import torch.nn.functional as F
import glob
import numpy as np
import pdb, os, argparse
from scipy import misc
import imageio
from model.CPD_ResNet_models import CPD_ResNet
from model.CPD_models import CPD_VGG
#from model.CPD_models import CPD_VGG
from model.CPD_MobileNet_models import CPD_MobileNet, CPD_MobileNet_Single
from data import test_dataset, get_loader
import time
from data_loader_bas import RescaleT, ToTensorLab, OtherTrans, SalObjDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from skimage import io, transform
from utils import clip_gradient, adjust_lr
from datetime import datetime
import sys
import warnings
import cv2
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=448, help='testing size')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--fake_rate', type=int, default=0.5, help='training batch size')

parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=str, default='mobile', help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.05, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=1, help='every n epochs decay learning rate')
opt = parser.parse_args()


dataset_path = '/home/hypevr/Desktop/demo/'#test_data/'
teach_path = dataset_path+'train/'

model_root = 'models/mobile_random_0.1/'#'models/CPD_ResNet50_scratch/'
model_dir = model_root + 'CPD_347.pth'

teach_model_dir = 'teacher.pt'
teach_model = torch.load(teach_model_dir)
teach_model.cuda()
teach_model.eval()

image_path = teach_path + 'image/'
mask_path = teach_path + 'mask/'
olay_path = teach_path + 'overlay/'
#fake_path = teach_path + 'overlay/'
back_path = '/home/hypevr/Desktop/data/projects/background/green/'


path, dirs, files = next(os.walk(image_path))
num_calib_images = len(files)

if not os.path.exists(mask_path):
    os.makedirs(mask_path)
if not os.path.exists(olay_path):
    os.makedirs(olay_path)

img_name_list = glob.glob(image_path + '*.jpg')

test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                    transform=transforms.Compose([RescaleT(opt.testsize), ToTensorLab(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

CE = torch.nn.BCEWithLogitsLoss()

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def overlay(image, mask):
    mask_3 = np.tile(np.expand_dims(mask, axis=-1), (1, 1, 3))
    olay = np.multiply(image, mask_3)
    return olay


def save_output(image_name, pred, d_dir, o_dir):
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()
    th = 0.2
    pred[pred > th] = 1
    pred[pred <= th] = 0

    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)

    mask = transform.resize(pred, (image.shape[0],image.shape[1]), anti_aliasing=False, mode = 'constant', order=0)
    mask = np.tile(np.expand_dims(mask, axis=-1), (1, 1, 3))
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    olay = image * mask

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    olay = olay.astype('uint8')
    #mask[mask<0.2] = 0
    mask = (mask*255).astype('uint8')
    io.imsave(o_dir + imidx + '.jpg', olay)
    io.imsave(d_dir + imidx + '.jpg', mask)


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

        # print(str(i) + '/' + str(total_step))
        # sys.stdout.write("\033[F")

        # if i == total_step:
        #     print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Dice1: {:0.4f}'.
        #           format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss.data, ))

    if epoch == opt.epoch:
        model.eval()
        torch.save(model.state_dict(), 'calib_weights.pth')


def test(test_model, test_loader, save_path, tr=0.3, after=False):
    inf_times = []
    for i in range(test_loader.size):
        image_orig, image, gt, name = test_loader.load_data()
        image = image.cuda()
        start_time = time.time()
        if after:
            res = test_model(image)
        else:
            _, res = test_model(image)
        torch.cuda.synchronize()
        inf_times.append(time.time() - start_time)
        res = F.upsample(res, size=(image_orig.shape[0], image_orig.shape[1]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #print(res)
        #input('wait')
        res[res < tr] = 0
        olay_mask = np.tile(np.expand_dims(res, axis=-1), (1, 1, 3))
        olay = image_orig * olay_mask
        #imageio.imwrite(save_path+name, res)
        olay = olay.astype('uint8')
        imageio.imwrite(save_path + name, olay)
    if after:
        print('average inference time: ' + str(sum(inf_times[3:])/(len(inf_times)-2)))


for i_test, data_test in enumerate(test_salobj_dataloader):
    inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)
    image_resized = inputs_test.numpy()[0, :, :, :].transpose((1, 2, 0))
    # io.imsave('after_resize.png', inputs_test.numpy()[0, :, :, :].transpose((1, 2, 0)))
    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    start = time.time()
    d1, d2, d3, d4, d5, d6, d7, d8 = teach_model(inputs_test)  # ,d2,d3,d4,d5,d6,d7,d8
    # print(d1)
    torch.cuda.synchronize()
    # d1 = d1.cpu()

    print('generating masks: ' + str(i_test) + '/' + str(num_calib_images))
    sys.stdout.write("\033[F")

    pred = normPRED(d1)
    # pred = overlay(image_resized, pred.squeeze().cpu().data.numpy())
    # save results to test_results folder
    save_output(img_name_list[i_test], pred, mask_path, olay_path)

print('mask generation finished!')

#model_dir = 'CPD.pth'
if opt.is_ResNet == 'resnet':
    model = CPD_ResNet()
    model.load_state_dict(torch.load(model_dir))
elif opt.is_ResNet == 'vgg':
    model = CPD_VGG()
    model.load_state_dict(torch.load(model_dir))
    #model.load_state_dict(torch.load('CPD.pth'))
elif opt.is_ResNet == 'mobile':
    model = CPD_MobileNet()
    model.load_state_dict(torch.load(model_dir))


# model = torch.load(model_dir)
model.cuda().eval()

test_img_dir = '/home/hypevr/Desktop/demo/test_fold/test_images/'
test_mask_dir = '/home/hypevr/Desktop/demo/test_fold/test_images/'

test_loader = test_dataset(test_img_dir, test_mask_dir, opt.testsize, True)
print('pre-testing with '+ str(test_loader.size) + ' images')
test(model, test_loader, '/home/hypevr/Desktop/demo/test_fold/before_calib/')

model.train()
train_loader = get_loader(image_path, mask_path, batchsize=opt.batchsize, trainsize=opt.trainsize, fake_back_rate=opt.fake_rate, back_dir=back_path)

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
for epoch in range(1, opt.epoch+1):
    print('calibrating epoch: '+ str(epoch) + '/' + str(opt.epoch))
    sys.stdout.write("\033[F")
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)

print('calibration finished!')
model = CPD_MobileNet_Single()
model.load_state_dict(torch.load('calib_weights.pth'))
model.eval().cuda()
scriptedmodel = torch.jit.script(model)
torch.jit.save(scriptedmodel, dataset_path + 'calib_model.pt')

model = torch.load(dataset_path + 'calib_model.pt')
model.cuda().eval()
print('after testing with '+ str(test_loader.size) + ' images')
test_loader = test_dataset(test_img_dir, test_mask_dir, opt.testsize, True)
test(model, test_loader, '/home/hypevr/Desktop/demo/test_fold/after_calib/', after=True)
print('all done')




