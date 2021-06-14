import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from scipy import misc
import imageio
from model.CPD_ResNet_models import CPD_ResNet
from model.CPD_models import CPD_VGG
#from model.CPD_models import CPD_VGG
from model.CPD_MobileNet_models import CPD_MobileNet, CPD_MobileNet_Single
from data import test_dataset
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=448, help='testing size')
parser.add_argument('--is_ResNet', type=str, default='mobile', help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_path = '/media/hypevr/KEY/'#test_data/'

model_root = 'models/mobile_random_0.1/'#'models/CPD_ResNet50_scratch/'
model_dir = model_root + 'CPD_143.pth'
#model_dir = 'CPD.pth'
if opt.is_ResNet == 'resnet':
    model = CPD_ResNet()
    model.load_state_dict(torch.load(model_dir))
elif opt.is_ResNet == 'vgg':
    model = CPD_VGG()
    model.load_state_dict(torch.load(model_dir))
    #model.load_state_dict(torch.load('CPD.pth'))
elif opt.is_ResNet == 'mobile':
    model = CPD_MobileNet_Single()
    model.load_state_dict(torch.load(model_dir))

model.cuda()
#model.eval()
#example = torch.rand(1, 3, 352, 352).cuda()
scriptedmodel = torch.jit.script(model)#, example)
torch.jit.save(scriptedmodel, 'model_save.pt')
model = torch.load('model_save.pt')
model.eval()

test_datasets = ['test_img'] #['test_images']

for dataset in test_datasets:
    # if opt.is_ResNet:
    #     save_path = dataset_path + 'test_results' + '/'
    # else:
    #     save_path = './results/VGG16/' + dataset + '/'
    save_path = dataset_path + dataset + '_masks_cpd_orig/'
    olay_path = dataset_path + dataset + '_olay_cpd_orig/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(olay_path):
        os.makedirs(olay_path)
    image_root = dataset_path + dataset +'/'
    gt_root = dataset_path + dataset +'/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize, True)
    for i in range(test_loader.size):
        image_orig, image, gt, name = test_loader.load_data()
        #gt = np.asarray(gt, np.float32)
        #gt /= (gt.max() + 1e-8)
        image = image.cuda()
        image = torch.tile(image, (1, 1, 1, 1))
        # print(image.shape)
        # input('wait')

        start_time = time.time()
        res = model(image) #
        torch.cuda.synchronize()
        res.cpu()
        print(time.time() - start_time)
        res = F.upsample(res, size=(image_orig.shape[0], image_orig.shape[1]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        res[res<0.5] = 0
        olay_mask = np.tile(np.expand_dims(res, axis=-1), (1, 1, 3))
        olay = image_orig * olay_mask

        imageio.imwrite(save_path+name, res)
        imageio.imwrite(olay_path + name, olay)
