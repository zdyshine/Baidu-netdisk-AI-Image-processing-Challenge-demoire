import random
import numpy as np
import dataloading.util as util
from paddle.io import Dataset
import paddle
import glob
import os

class Task1_Traindataset(Dataset):
    def __init__(self, opt, mode='train'):
        super(Task1_Traindataset, self).__init__()
        # print(os.path.join(opt.train_root, 'gt', '*.png'))
        if mode == 'train':
            self.paths_GT = sorted(glob.glob(os.path.join(opt.train_root, 'gts', '*.jpg'))) * 10
            self.paths_LQ = sorted(glob.glob(os.path.join(opt.train_root, 'images', '*.jpg'))) * 10
        elif mode == 'val':
            self.paths_GT = sorted(glob.glob(os.path.join(opt.val_root, 'gts', '*.jpg')))
            self.paths_LQ = sorted(glob.glob(os.path.join(opt.val_root, 'images', '*.jpg')))
            # self.paths_GT, self.paths_LQ = self.paths_GT[::50], self.paths_LQ[::50]
            # print(self.paths_GT, self.paths_LQ)
        else:
            print('mode is train or val')

        print('=========> Total train/val images: {}.'.format(len(self.paths_GT)))
        assert self.paths_GT, 'Error: GT path is empty.'
        # if self.paths_LQ and self.paths_GT:
        assert len(self.paths_LQ) == len(
            self.paths_GT), 'GT and LQ datasets have different number of images - {}, {}.'.format(
            len(self.paths_LQ), len(self.paths_GT))
        self.opt = opt
        self.mode = mode

    def __getitem__(self, index):
        GT_path, LQ_path = None, None

        # get GT image
        GT_path = self.paths_GT[index]
        LQ_path = self.paths_GT[index].replace('gts', 'images')
        img_GT = util.read_img(GT_path)
        img_LQ = util.read_img(LQ_path)

        # print(img_LQ.shape, GT_path.replace('HR', 'LR'), img_GT.shape, GT_path)
        H, W, C = img_LQ.shape
        H_gt, W_gt, C = img_GT.shape
        if H != H_gt:
            print('*******wrong image*******:{}'.format(LQ_path))

        # randomly crop
        if self.mode == 'train':
            if self.opt.patch_size is not None:
                LQ_size = self.opt.patch_size
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                img_GT = img_GT[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt.use_flip, self.opt.use_rot)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]

        img_GT, img_LQ = img_GT.astype(np.float32), img_LQ.astype(np.float32)
        img_GT = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1))))
        img_LQ = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1))))

        # print(img_LQ.shape, GT_path.replace('HR', 'LR'), img_GT.shape, GT_path)

        return img_LQ, img_GT  # , LQ_path, GT_path

    def __len__(self):
        return len(self.paths_GT)
