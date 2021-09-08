import torch.utils as utils
import torch
import numpy as np
import random
import cv2
import os
import scipy.io as sio
from torch.utils import data
from imageio import imread
import torchvision.transforms as transforms
import flow_transforms

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]


def noisy(noise_typ, img):
    if noise_typ == "gauss":
        mean = 0
        var = 10

        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (460, 620))

        noisy_image = np.zeros(img.shape, np.float32)

        if len(img.shape) == 2:
            noisy_image = img + gaussian
        else:
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        return noisy_image

    elif noise_typ == "s&p":
        prob = 0.1
        output = np.zeros(img.shape, np.uint8)
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output

    elif noise_typ == "compress":
        img = cv2.imread(img, 0)
        cv2.imwrite('tmp.jpg', img, encode_param)
        output = cv2.imread('tmp.jpg', 0)
        os.remove('tmp.jpg')
        return output


class PIR_Dataset(utils.data.Dataset):
    def __init__(self, path_source, path_target, path_lables, noise):

        self.path_source = path_source
        self.image_path = []

        self.path_target = path_target
        self.target_path = []

        self.path_lables = path_lables
        self.lab_path = []

        self.noise = noise

        scr_files = os.listdir(path_source)
        scr_files.sort(key=lambda x: int(x[:-4][5:]))
        for file in scr_files:
            self.image_path.append(file)
        print("source_image_numbers: ", len(self.image_path))
        # print(self.image_path)

        lab_files = os.listdir(path_lables)
        lab_files.sort(key=lambda x: int(x[:-4]))
        for file in lab_files:
            self.lab_path.append(file)
        print("lable_image_numbers: ", len(self.lab_path))
        # print(self.lab_path)

        tar_files = os.listdir(path_target)
        tar_files.sort(key=lambda x: int(x[:-6]))
        for file in tar_files:
            self.target_path.append(file)
        print("target_image_numbers: ", len(self.target_path))
        # print(self.target_path)



    def __getitem__(self, index):
        data = sio.loadmat(self.path_lables + '/' + str(self.lab_path[index]))
        fmap = data['fmap']
        img = imread(self.path_source + '/' + str(self.image_path[index]))
        tar = imread(self.path_target + '/' + str(self.target_path[index]))


        if self.noise == 'Vanilla':
            img = img
            tar = tar
        elif self.noise == 'Blur5':
            img = cv2.blur(img, (5, 5))
        elif self.noise == 'Blur10':
            img = cv2.blur(img, (10, 10))
        elif self.noise == 'S&P':
            img = noisy('s&p', img)
        elif self.noise == 'Gaussian':
            img = noisy('gauss', img)
            tar = noisy('gauss', tar)
        elif self.noise == 'Compression':
            img = noisy('compress', self.path_source + '/' + str(self.image_path[index]))

        # Data loading code
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.45, 0.432, 0.411], std=[1, 1, 1])
        ])


        target_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[20, 20, 1])
        ])

        img1 = input_transform(img)
        img2 = input_transform(tar)
        input_var = torch.cat([img1, img2])
        fmap = target_transform(fmap)

        datum = (input_var, fmap)
        return datum

    def __len__(self):
        return len(self.image_path)

