import argparse
import numpy as np
import os
from typing import Tuple

import torch
from nilearn import plotting
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils.helper import load_dict
from utils.helper import saveasnii
from utils.ols import OLS_pytorch
from utils.ols import vectorized_correlation


def build_argparser() -> argparse:
    """ Build the args parser

    :return:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Encoding model analysis for Algonauts 2021')
    parser.add_argument('-rd', '--result_dir',
                        help='saves predicted fMRI activity',
                        default='./results',
                        type=str)
    parser.add_argument('-ad', '--activation_dir',
                        help='directory containing DNN activations',
                        default='./features/',
                        type=str)
    parser.add_argument('-model', '--model',
                        help='model name under which predicted fMRI activity will be saved',
                        default='alexnet_devkit',
                        type=str)
    parser.add_argument('-l', '--layer',
                        help='layer from which activations will be used to train and predict fMRI activity',
                        default='layer_5',
                        type=str)
    parser.add_argument('-sub', '--sub',
                        help='subject number from which real fMRI data will be used',
                        default=4,
                        type=int)
    parser.add_argument('-r', '--roi',
                        help='brain region, from which real fMRI data will be used',
                        default='EBA',
                        type=str)
    parser.add_argument('-m', '--mode',
                        help='test or val, val returns mean correlation by using 10% of training data for validation',
                        default='val',
                        type=str)
    parser.add_argument('-fd', '--fmri_dir',
                        help='directory containing fMRI activity',
                        default='./participants_data_v2021',
                        type=str)
    parser.add_argument('-v', '--visualize',
                        help='visualize whole brain results in MNI space or not',
                        default=True,
                        type=bool)
    parser.add_argument('-b', '--batch_size',
                        help=' number of voxel to fit at one time in case of memory constraints',
                        default=1000,
                        type=int)
    parser.add_argument('-ft', '--features_type',
                        help='Type of feature representation',
                        default='pca_100',
                        type=str)
    parser.add_argument('-s', '--solver',
                        help='Type of solver to use (sklearn_lr or ols)',
                        default='sklearn_lr',
                        type=str)

    return parser


def get_activations(activations_dir: str) -> Tuple[np.array, np.array]:
    """This function loads neural network features/activations (preprocessed using PCA) into a
    numpy array according to a given layer.

    :param activations_dir:
        Path to PCA processed Neural Network features

    :return train_activations:
        matrix of dimensions #train_vids x #pca_components
        containing activations of train videos
    :return test_activations:
        matrix of dimensions #test_vids x #pca_components
        containing activations of test videos
    """

    train_file = os.path.join(activations_dir, "train.npy")
    test_file = os.path.join(activations_dir, "test.npy")

    train_activations = np.load(train_file)
    test_activations = np.load(test_file)

    scaler = StandardScaler()
    train_activations = scaler.fit_transform(train_activations)
    test_activations = scaler.fit_transform(test_activations)

    return train_activations, test_activations


def get_fmri(fmri_dir: str, ROI: str) -> np.array:
    """This function loads fMRI data into a numpy array for to a given ROI.

    :param fmri_dir:
        path to fMRI data directory
    :param ROI:
        Name of region of interest

    :return:
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI
    """

    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions
    ROI_data_train = np.mean(ROI_data["train"], axis=1)
    if ROI == "WB":
        voxel_mask = ROI_data['voxel_mask']
        return ROI_data_train, voxel_mask

    return ROI_data_train


def predict_fmri_fast(train_activations: np.array, test_activations: np.array, train_fmri: np.array,
                      solver: str, use_gpu: bool = False) -> np.array:
    """This function fits a linear regression using train_activations and train_fmri,
    then returns the predicted fmri_pred_test using the fitted weights and
    test_activations.

    :param train_activations:
        matrix of dimensions #train_vids x #pca_components
        containing activations of train videos.
    :param test_activations:
        matrix of dimensions #test_vids x #pca_components
        containing activations of test videos
    :param train_fmri:
        matrix of dimensions #train_vids x  #voxels
        containing fMRI responses to train videos
    :param solver:
        Solver to use for the linear regression
    :param use_gpu:
        Description of parameter `use_gpu`.

    :return fmri_pred_test:
        matrix of dimensions #test_vids x  #voxels
        containing predicted fMRI responses to test videos .
    """

    if solver == 'sklearn_lr':
        reg = LinearRegression(n_jobs=-1).fit(train_activations, train_fmri)
        fmri_pred_test = reg.predict(test_activations)
    elif solver == 'ols':
        # Default provided with the dev kit --> out of memory issue common
        reg = OLS_pytorch(use_gpu)
        reg.fit(train_activations, train_fmri.T)
        fmri_pred_test = reg.predict(test_activations)
    else:
        raise ValueError('Incorrect solver type. Must be "sklearn_lr" or "ols"')

    return fmri_pred_test


def main(args):

    mode = args.mode  # test or val
    sub = f"sub{args.sub:02}"
    ROI = args.roi
    model = args.model
    layer = args.layer
    visualize_results = args.visualize
    batch_size = args.batch_size  # number of voxel to fit at one time in case of memory constraints
    features_type = args.features_type
    solver = args.solver

    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    if ROI == "WB":
        track = "full_track"
    else:
        track = "mini_track"

    activation_dir = os.path.join(args.activation_dir, model, layer, features_type)
    fmri_dir = os.path.join(args.fmri_dir, track)

    sub_fmri_dir = os.path.join(fmri_dir, sub)
    results_dir = os.path.join(args.result_dir, model, layer, features_type, track, sub)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("ROi is : ", ROI)

    train_activations, test_activations = get_activations(activation_dir)
    if track == "full_track":
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir, ROI)
    num_voxels = fmri_train_all.shape[1]

    if mode == 'val':
        # Here as an example we use first 900 videos as training and rest of the videos as validation
        test_activations = train_activations[900:, :]
        train_activations = train_activations[:900, :]

        fmri_train = fmri_train_all[:900, :]
        fmri_test = fmri_train_all[900:, :]

        pred_fmri = np.zeros_like(fmri_test)
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_val.npy')
    else:
        fmri_train = fmri_train_all
        num_test_videos = 102

        pred_fmri = np.zeros((num_test_videos, num_voxels))
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_test.npy')

    print("number of voxels is ", num_voxels)

    ii = 0
    while ii < num_voxels - batch_size:
        pred_fmri[:, ii:ii + batch_size] = predict_fmri_fast(train_activations, test_activations,
                                                             fmri_train[:, ii:ii + batch_size],
                                                             use_gpu=use_gpu, solver=solver)
        ii = ii + batch_size
        print((100 * ii) // num_voxels, " percent complete")

    pred_fmri[:, ii:] = predict_fmri_fast(train_activations, test_activations, fmri_train[:, ii:ii + batch_size],
                                          use_gpu=use_gpu, solver=solver)

    if mode == 'val':
        score = vectorized_correlation(fmri_test, pred_fmri)
        print("----------------------------------------------------------------------------")
        print("Mean correlation for ROI : ", ROI, "in ", sub, " is :", round(score.mean(), 3))

        # result visualization for whole brain (full_track)
        if track == "full_track" and visualize_results:
            visual_mask_3D = np.zeros((78, 93, 71))
            visual_mask_3D[voxel_mask == 1] = score
            brain_mask = './example.nii'

            nii_save_path = os.path.join(results_dir, ROI + '_val.nii')
            saveasnii(brain_mask, nii_save_path, visual_mask_3D)

            view = plotting.view_img_on_surf(nii_save_path, threshold=None, surf_mesh='fsaverage',
                                             title='Correlation for sub' + sub, colorbar=False)

            view_save_path = os.path.join(results_dir, ROI + '_val.html')
            view.save_as_html(view_save_path)
            print("Results saved in this directory: ", results_dir)
            view.open_in_browser()

    np.save(pred_fmri_save_path, pred_fmri)

    print("----------------------------------------------------------------------------")
    print("ROI done : ", ROI)


if __name__ == "__main__":
    main(build_argparser().parse_args())
