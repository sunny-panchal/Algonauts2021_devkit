###
# This file will:
# 1. Generate and save Alexnet features in a given folder
# 2. preprocess Alexnet features using PCA and save them in another folder
###
import argparse
import glob
import numpy as np
import os
import random
import urllib
from tqdm import tqdm
from typing import List
from typing import Tuple

import cv2
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms as trn
from torch.autograd import Variable as V

from alexnet import *


seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)


def get_video_from_mp4(file: str, sampling_rate: int) -> Tuple[np.array, int]:
    """This function takes a mp4 video file as input and returns
    an array of frames in numpy format.

    :param file:
        path to mp4 video file
    :param sampling_rate:
        how many frames to skip when doing frame sampling.

    :return video:
    :return num_frames:
        number of frames extracted
    """
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((int(frameCount / sampling_rate), frameHeight,
                    frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        fc += 1
        if fc % sampling_rate == 0:
            (ret, buf[int((fc - 1) / sampling_rate)]) = cap.read()

    cap.release()
    return np.expand_dims(buf, axis=0), int(frameCount / sampling_rate)


def load_alexnet(model_checkpoints: str) -> AlexNet:
    """This function initializes an Alexnet and load its weights from a pretrained model

    :param model_checkpoints:
        model checkpoints location.

    :return model:
        pytorch model of alexnet
    """

    model = alexnet()
    model_file = model_checkpoints
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    model_dict = ["conv1.0.weight", "conv1.0.bias", "conv2.0.weight", "conv2.0.bias", "conv3.0.weight", "conv3.0.bias",
                  "conv4.0.weight", "conv4.0.bias", "conv5.0.weight", "conv5.0.bias", "fc6.1.weight", "fc6.1.bias",
                  "fc7.1.weight", "fc7.1.bias", "fc8.1.weight", "fc8.1.bias"]
    state_dict = {}
    i = 0
    for k, v in checkpoint.items():
        state_dict[model_dict[i]] = v
        i += 1

    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def get_activations_and_save(model: AlexNet, video_list: List, activations_dir: str, sampling_rate: int = 4):
    """This function generates Alexnet features and save them in a specified directory.

    :param model:
        pytorch model : alexnet.
    :param video_list:
        the list contains path to all videos.
    :param activations_dir:
        save path for extracted features.
    :param sampling_rate:
        how many frames to skip when feeding into the network.
    """

    centre_crop = trn.Compose([
        trn.ToPILImage(),
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for video_file in tqdm(video_list):
        vid, num_frames = get_video_from_mp4(video_file, sampling_rate)
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        activations = []
        for frame in range(num_frames):
            img = vid[0, frame, :, :, :]
            input_img = V(centre_crop(img).unsqueeze(0))
            if torch.cuda.is_available():
                input_img = input_img.cuda()
            x = model.forward(input_img)
            for i, feat in enumerate(x):
                if frame == 0:
                    activations.append(feat.data.cpu().numpy().ravel())
                else:
                    activations[i] = activations[i] + feat.data.cpu().numpy().ravel()

        for layer in range(len(activations)):
            save_path = os.path.join(activations_dir, f'layer_{layer + 1}', 'full')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_path = os.path.join(save_path, f'{video_file_name}.npy')

            np.save(save_path, activations[layer] / float(num_frames))


def do_PCA_and_save(activations_dir: str, n_components: int):
    """This function preprocesses Neural Network features using PCA and save the results
    in  a specified directory
.
    :param activations_dir:
        save path for extracted features.
    :param n_components:
        Number of components for PCA
    """

    layers = ['layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5', 'layer_6', 'layer_7', 'layer_8']

    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir + f'/{layer}/full/*.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0])
        x = np.zeros((len(activations_file_list), feature_dim.shape[0]))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i, :] = temp
        x_train = x[:1000, :]
        x_test = x[1000:, :]

        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        ipca = PCA(n_components=n_components)  # , batch_size=20)
        ipca.fit(x_train)

        x_train = ipca.transform(x_train)
        x_test = ipca.transform(x_test)

        pca_save_path = os.path.join(activations_dir, layer, f"pca_{n_components}")
        if not os.path.exists(pca_save_path):
            os.makedirs(pca_save_path)

        train_save_path = os.path.join(pca_save_path, 'train.npy')
        test_save_path = os.path.join(pca_save_path, 'test.npy')
        np.save(train_save_path, x_train)
        np.save(test_save_path, x_test)


def main(args: argparse.Namespace):
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_dir = args.video_data_dir
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))

    # load Alexnet
    # Download pretrained Alexnet from:
    # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
    # and save in the current directory
    checkpoint_path = "./alexnet.pth"
    if not os.path.exists(checkpoint_path):
        url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
        urllib.request.urlretrieve(url, "./alexnet.pth")
    model = load_alexnet(checkpoint_path)

    # get and save activations
    print("-------------Saving activations ----------------------------")
    get_activations_and_save(model, video_list, save_dir)

    # preprocessing using PCA and save
    print("-------------performing  PCA----------------------------")
    do_PCA_and_save(save_dir, n_components=args.pca)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extraction from Alexnet and preprocessing using PCA')
    parser.add_argument('-vdir', '--video_data_dir',
                        help='video data directory',
                        default='./AlgonautsVideos268_All_30fpsmax/',
                        type=str)
    parser.add_argument('-sdir', '--save_dir',
                        help='saves processed features',
                        default='./features/alexnet_devkit',
                        type=str)
    parser.add_argument('-pca', '--pca',
                        help='PCA dimension size',
                        default=100,
                        type=int)

    main(parser.parse_args())
