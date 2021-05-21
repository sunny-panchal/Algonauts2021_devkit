###
# This file will:
# 1. Generate and save EfficientNet features in a given folder from the Sense repository
# 2. preprocess sense features using PCA and save them in another folder
###
import argparse
import glob
import numpy as np
import os
import random
import torch
from typing import List

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sense import engine
from sense.backbone_networks import StridedInflatedEfficientNet
from sense.finetuning import compute_features
from sense.finetuning import extract_frames
from sense.finetuning import MODEL_TEMPORAL_DEPENDENCY
from sense.loading import load_weights


seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    use_gpu = True
else:
    use_gpu = False


def load_efficientnet() -> StridedInflatedEfficientNet:
    """This function initializes an EfficientNet from Sense and loads
    its weights from a pretrained model

    :return model:
        pytorch model of efficientnet

    """
    model = StridedInflatedEfficientNet()
    model_weights = load_weights('../sense/resources/backbone/strided_inflated_efficientnet.ckpt')
    model.load_state_dict(model_weights)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def get_activations_and_save(model: StridedInflatedEfficientNet, video_list: List, activations_dir: str):
    """This function generates EfficientNet features and save them in a specified directory.

    :param model:
        Pytorch model of SI-EN from Sense
    :param video_list:
        List of all videos
    :param activations_dir:
        Path to save extracted features
    """

    # Create inference engine
    inference_engine = engine.InferenceEngine(model, use_gpu=use_gpu)

    # extract features
    num_videos = len(video_list)
    for video_index, video_path in enumerate(video_list):
        video_file_name = os.path.split(video_path)[-1].split(".")[0]
        print(f'\rExtract features from video {video_index + 1} / {num_videos}')
        path_features = os.path.join(activations_dir, 'full', f'{video_file_name}.npy')

        if os.path.isfile(path_features):
            print("\tSkipped - feature was already precomputed.")
        else:
            # Read all frames
            frames = extract_frames(video_path=video_path,
                                    inference_engine=inference_engine)
            compute_features(path_features=path_features,
                             inference_engine=inference_engine,
                             frames=frames,
                             batch_size=16)

        print('\n')


def load_efficientnet_features(features_path: str) -> np.ndarray:
    """Load and pre-process efficientnet features

    :param features_path:
        Path to predicted features

    :return:
        Array of processed features
    """
    features = np.load(features_path)
    num_preds = features.shape[0]
    num_timesteps = 1
    stride = 4
    num_frames_padded = int((MODEL_TEMPORAL_DEPENDENCY - 1) / stride)

    # remove padded frames
    minimum_position = min(num_preds - num_timesteps - 1,
                           num_frames_padded)
    minimum_position = max(minimum_position, 0)
    position = np.random.randint(minimum_position, num_preds - num_timesteps)
    return features[position: position + num_timesteps].flatten()


def do_PCA_and_save(activations_dir: str, n_components: int):
    """This function preprocesses Neural Network features using PCA and save the results
    in  a specified directory
.
    :param activations_dir:
        save path for extracted features.
    :param n_components:
        PCA dimension size
    """

    activations_file_list = glob.glob(activations_dir + '/full/*.npy')
    activations_file_list.sort()
    feature_dim = load_efficientnet_features(activations_file_list[0])
    x = np.zeros((len(activations_file_list), feature_dim.shape[0]))

    for i, activation_file in enumerate(activations_file_list):
        temp = load_efficientnet_features(activation_file)
        x[i, :] = temp

    x_train = x[:1000, :]
    x_test = x[1000:, :]

    x_test = StandardScaler().fit_transform(x_test)
    x_train = StandardScaler().fit_transform(x_train)
    ipca = PCA(n_components=n_components)
    ipca.fit(x_train)

    x_train = ipca.transform(x_train)
    x_test = ipca.transform(x_test)

    pca_save_path = os.path.join(activations_dir, f"pca_{n_components}")
    if not os.path.exists(pca_save_path):
        os.makedirs(pca_save_path)

    train_save_path = os.path.join(pca_save_path, 'train.npy')
    test_save_path = os.path.join(pca_save_path, 'test.npy')
    np.save(train_save_path, x_train)
    np.save(test_save_path, x_test)


def main(args):
    n_components = args.pca
    save_dir = os.path.join(args.save_dir, 'layer_n')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_dir = args.video_data_dir
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))

    # load pre-trained EfficientNet
    model = load_efficientnet()

    # get and save activations
    print("-------------Saving activations ----------------------------")
    get_activations_and_save(model, video_list, save_dir)

    # preprocessing using PCA and save
    print("-------------performing  PCA----------------------------")
    do_PCA_and_save(save_dir, n_components)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extraction from EfficientNet and preprocessing using PCA')

    parser.add_argument('-vdir', '--video_data_dir',
                        help='video data directory',
                        default='./AlgonautsVideos268_All_30fpsmax/',
                        type=str)
    parser.add_argument('-sdir', '--save_dir',
                        help='saves processed features',
                        default='./features/sien_sense',
                        type=str)
    parser.add_argument('-pca', '--pca',
                        help='PCA dimension size',
                        default=100,
                        type=int)

    main(parser.parse_args())
