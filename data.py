import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from skimage import io, transform, color
import random
import torch
import torchvision.transforms.functional as TF
import sys
import cv2
from data_loader_bas import OtherTrans


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, fake_back_rate=0, back_dir=None):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gt_root = gt_root
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.resize = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])
        self.img_transform_after = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform_after = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        self.fake_back_rate = fake_back_rate
        self.fb = FakeBack(back_dir, trainsize=self.trainsize)

    def __getitem__(self, index):
        filename = self.images[index].split('/')[-1]
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gt_root + filename)
        #image = self.resize(image)
        #gt = self.resize(gt)

        if self.fake_back_rate:
            if random.random() < self.fake_back_rate:
                sample = {'image': image, 'label': gt}
                sample = self.fb(sample)
                image = sample['image']
                gt = sample['label']
            else:
                image = self.resize(image)
                gt = self.resize(gt)
        else:
            image = self.resize(image)
            gt = self.resize(gt)

        image = self.img_transform_after(image)
        #gt = self.gt_transform_after(gt)
        image = image.float()
        gt = gt.float()
        return image, gt

    def filter_files(self):
        #print(len(self.images), len(self.gts))
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True, fake_back_rate=0, back_dir=None):

    dataset = SalObjDataset(image_root, gt_root, trainsize, fake_back_rate=fake_back_rate, back_dir=back_dir)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize, orig=False):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
        self.orig = orig

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        if self.orig:
            image_orig = image.copy()
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        if self.orig:
            return np.asarray(image_orig), image, gt, name
        else:
            return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

class FakeBack(object):

    def __init__(self, back_dir, trainsize):
        self.back_dir = back_dir
        self.trainsize = trainsize
        if self.back_dir:
            path, dirs, files = next(os.walk(back_dir))
            random.shuffle(files)
            num = len(files)
            self.selected_backs = files[:int(num*1.)]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        trans_r = 0.2
        # erode = random.randint(1, 8)
        # kernel = np.ones((3, 3), np.uint8)
        # label = cv2.erode(label, kernel, iterations=erode)
        rand_shift = int(self.trainsize * trans_r)
        zoom = random.uniform(0.8, 1.25)
        angle = random.randint(-20, 20)
        translate = [random.randint(-rand_shift, rand_shift),
                     random.randint(-rand_shift, rand_shift)]

        zoom_size = int(self.trainsize * zoom)
        if zoom_size%2 == 1:
            zoom_size += 1
        image = image.resize((zoom_size, zoom_size))
        label = label.resize((zoom_size, zoom_size))


        if zoom > 1:
            image = TF.center_crop(image, [self.trainsize, self.trainsize])
            label = TF.center_crop(label, [self.trainsize, self.trainsize])
        else:
            pad = int((self.trainsize - zoom_size)/2)
            image = TF.pad(image, padding=[pad, pad, pad, pad])
            label = TF.pad(label, padding=[pad, pad, pad, pad])

        image = TF.affine(image, angle=angle, translate=translate, fill=0, shear=[0, 0], scale=1)
        label = TF.affine(label, angle=angle, translate=translate, fill=0, shear=[0, 0], scale=1)

        back = Image.open(self.back_dir+random.choice(self.selected_backs))
        back = back.resize((self.trainsize, self.trainsize))
        back = np.asarray(back)
        #io.imsave('back.jpg', back)

        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        image = np.asarray(image)
        label = np.asarray(label)
        erode = random.randint(1, 5)
        kernel = np.ones((3, 3), np.uint8)
        label = cv2.erode(label, kernel, iterations=erode)

        label_cp = label.copy()
        label_cp = 255. * (label_cp.astype(np.float32)/float(label.max()+1e-8))
        label_cp[label < 10] = 0
        image = image*(np.tile(np.expand_dims(label_cp, axis=-1), (1, 1, 3))/255.)
        #kernel = np.ones((3, 3), np.uint8)
        #image = cv2.erode(image, kernel, iterations=5)

        olay = image.copy()
        compare = np.all(image == (0, 0, 0), axis=-1)
        olay[compare] = back[compare]
        #olay.astype(np.float32)
        #label_cp.astype(np.float32)
        # io.imsave('fake.jpg', olay)
        # io.imsave('fake_label.jpg', label_cp)
        # #input('wait')
        # sys.exit()
        olay = olay/255.
        label_cp = np.expand_dims(label_cp/255., axis=0)
        olay = np.transpose(olay, (2, 0, 1))

        olay = torch.tensor(olay)
        label_cp = torch.tensor(label_cp)

        #FLP = transforms.RandomHorizontalFlip()
        #RPe = transforms.RandomPerspective(distortion_scale=0.1, p=0.5)
        #RRo = transforms.RandomRotation(90)

        return {'image': olay, 'label': label_cp}


