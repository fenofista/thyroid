# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2024/12/12 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import torch
from scipy import stats

from lib.utils import *
import lib.transforms.two as my_transforms
from lib import utils, dataloaders, models, metrics, testers

params_3D_CBCT_Tooth = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [0.5, 0.5, 0.5],
    "clip_lower_bound": -1412,
    "clip_upper_bound": 17943,
    "samples_train": 2048,
    "crop_size": (160, 160, 96),
    "crop_threshold": 0.5,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.3,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 20,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "3D-CBCT-Tooth",
    "dataset_path": r"./datasets/3D-CBCT-Tooth",
    "create_data": False,
    "batch_size": 1,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 1,
    "classes": 2,
    "scaling_version": "TINY",
    "dimension": "3d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "ReduceLROnPlateau",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.00551122, 0.99448878],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 20,
    "best_dice": 0,
    "update_weight_freq": 32,
    "terminal_show_freq": 256,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_DRIVE = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.16155026, 0.26819696, 0.50784565),
    "normalize_stds": (0.10571646, 0.18532471, 0.35080457),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "DRIVE",
    "dataset_path": r"./datasets/DRIVE",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict": {0: "background", 1: "foreground"},
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.08631576554733908, 0.913684234452661],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 5,
    "save_epoch_freq": 50,
}

params_STARE = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.11336552, 0.33381058, 0.58892505),
    "normalize_stds": (0.10905356, 0.19210595, 0.35295892),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "STARE",
    "dataset_path": r"./datasets/STARE",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict": {0: "background", 1: "foreground"},
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.07542384887839432, 0.9245761511216056],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 2,
    "save_epoch_freq": 50,
}

params_CHASE_DB1 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (512, 512),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.31154673834507396,
    "color_jitter": 0.20193812578236442,
    "random_rotation_angle": 45,
    "normalize_means": (0.02789665, 0.16392259, 0.45287978),
    "normalize_stds": (0.03700363, 0.14539037, 0.36542216),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "CHASE-DB1",
    "dataset_path": r"./datasets/CHASE-DB1",
    "batch_size": 2,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict": {0: "background", 1: "foreground"},
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.001,
    "momentum": 0.705859940948433,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9251490005593288,
    "step_size": 10,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 50,
    "T_0": 10,
    "T_mult": 4,
    "mode": "max",
    "patience": 5,
    "factor": 0.8,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.07186707540874207, 0.928132924591258],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 200,
    "best_metric": 0,
    "terminal_show_freq": 5,
    "save_epoch_freq": 50,
}

params_Kvasir_SEG = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.22448543324157222,
    "color_jitter": 0.3281010563062837,
    "random_rotation_angle": 30,
    "normalize_means": (0.24398195, 0.32772844, 0.56273),
    "normalize_stds": (0.18945072, 0.2217485, 0.31491405),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "Kvasir-SEG",
    "dataset_path": r"./datasets/Kvasir-SEG",
    "batch_size": 1,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "scaling_version": "BASIC",
    "dimension": "2d",
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "Adam",
    "learning_rate": 0.0005,
    "weight_decay": 0.000001,
    "momentum": 0.7781834740942233,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8079569870480704,
    "step_size": 20,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 200,
    "T_0": 10,
    "T_mult": 2,
    "mode": "max",
    "patience": 5,
    "factor": 0.91,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.1557906849111095, 0.8442093150888904],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 400,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 150,
}


def segment_image(params, model, image, label):
    transform = my_transforms.Compose(
        [
            my_transforms.Resize(params["resize_shape"]),
            my_transforms.ToTensor(),
            my_transforms.Normalize(
                mean=params["normalize_means"], std=params["normalize_stds"]
            ),
        ]
    )
    label[label == 255] = 1
    # 数据预处理和数据增强
    image, label = transform(image, label)
    # image扩充一维
    image = torch.unsqueeze(image, dim=0)
    # 转换数据格式
    label = label.to(dtype=torch.uint8)
    # 预测分割
    pred = torch.squeeze(model(image.to(params["device"])), dim=0)
    segmented_image_np = torch.argmax(pred, dim=0).to(dtype=torch.uint8).cpu().numpy()
    label_np = label.numpy()
    # image和numpy扩展到三维
    seg_image = np.dstack([segmented_image_np] * 3)
    label = np.dstack([label_np] * 3)
    # 定义红色、白色和绿色图像
    red = np.zeros((224, 224, 3))
    red[:, :, 0] = 255
    green = np.zeros((224, 224, 3))
    green[:, :, 1] = 255
    white = np.ones((224, 224, 3)) * 255
    segmented_display_image = np.zeros((224, 224, 3))
    segmented_display_image = np.where(
        seg_image & label, white, segmented_display_image
    )
    segmented_display_image = np.where(seg_image & ~label, red, segmented_display_image)
    segmented_display_image = np.where(
        ~seg_image & label, green, segmented_display_image
    )
    JI_score = cal_jaccard_index(seg_image, label)
    return segmented_display_image, JI_score


def get_analyse_image_and_dsc(seg_image, label):
    label[label == 255] = 1
    dsc_score = utils.cal_dsc(seg_image, label)
    seg_image = np.dstack([seg_image] * 3)
    label = np.dstack([label] * 3)
    red = np.zeros_like(seg_image)
    red[:, :, 0] = 255
    green = np.zeros_like(seg_image)
    green[:, :, 1] = 255
    white = np.ones_like(seg_image) * 255
    segmented_display_image = np.zeros_like(seg_image)
    segmented_display_image = np.where(seg_image & label, white, segmented_display_image)
    segmented_display_image = np.where(seg_image & ~label, red, segmented_display_image)
    segmented_display_image = np.where(~seg_image & label, green, segmented_display_image)
    return segmented_display_image, dsc_score


def generate_segment_result_images(
    dataset_name, model_names, seed=1777777, benchmark=False, deterministic=True
):
    # build the dir to save the result
    result_dir = os.path.join(r"./images", dataset_name + "_segment_result")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    # select the dictionary of hyperparameters used for training
    if dataset_name == "DRIVE":
        params = params_DRIVE
    elif dataset_name == "STARE":
        params = params_STARE
    elif dataset_name == "CHASE-DB1":
        params = params_CHASE_DB1
    elif dataset_name == "Kvasir-SEG":
        params = params_Kvasir_SEG
    else:
        raise RuntimeError(f"No {dataset_name} dataset available")
    # launch initialization
    utils.reproducibility(seed, deterministic, benchmark)
    # define the dataset path
    dataset_root_dir = os.path.join(r"./datasets", dataset_name, "test")
    images_dir = os.path.join(dataset_root_dir, "images")
    labels_dir = os.path.join(dataset_root_dir, "annotations")
    cnt = 0
    # traverse each images
    for image_name in tqdm(os.listdir(images_dir)):
        filename, ext = os.path.splitext(image_name)
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, filename + ".png")
        # load image
        image = cv2.imread(image_path, -1)
        label = cv2.imread(label_path, -1)
        max_JI_score = 0
        max_model_name = None
        segment_result_images_list = []

        # traverse each models
        for model_name in model_names:
            params["model_name"] = model_name
            params["pretrain"] = os.path.join(
                r"./pretrain", dataset_name + "_" + model_name + ".pth"
            )
            # initialize model
            model = models.get_model(params)
            # load model weight
            pretrain_state_dict = torch.load(
                params["pretrain"],
                map_location=lambda storage, loc: storage.cuda(params["device"]),
            )
            model_state_dict = model.state_dict()
            for param_name in model_state_dict.keys():
                if (param_name in pretrain_state_dict) and (
                    model_state_dict[param_name].size()
                    == pretrain_state_dict[param_name].size()
                ):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
            model.load_state_dict(model_state_dict, strict=True)
            # segment
            seg_result_image, JI_score = segment_image(
                params, model, image.copy(), label.copy()
            )
            # save the segmented images
            segment_result_images_list.append(seg_result_image)
            # update max JI metric
            if JI_score > max_JI_score:
                max_JI_score = JI_score
                max_model_name = model_name
        # meet the conditions for preservation
        if max_model_name == "PMFSNet":
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                os.path.join(
                    result_dir,
                    "{:04d}".format(cnt) + "_0.jpg",
                ),
                image,
            )
            label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                os.path.join(
                    result_dir,
                    "{:04d}".format(cnt) + "_1.jpg",
                ),
                label,
            )
            for j, segment_result_image in enumerate(segment_result_images_list):
                segment_result_image = cv2.resize(segment_result_image, (224, 224), interpolation=cv2.INTER_NEAREST)
                segment_result_image = segment_result_image[:, :, ::-1]
                cv2.imwrite(
                    os.path.join(
                        result_dir,
                        "{:04d}".format(cnt) + "_" + str(j + 2) + ".jpg",
                    ),
                    segment_result_image,
                )
            cnt += 1


def generate_tooth_segment_result_images(model_names):
    params = params_3D_CBCT_Tooth
    # build the dir to save the result
    result_dir = os.path.join(r"./images", params["dataset_name"] + "_segment_result")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)
    # launch initialization
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])
    # define the dataset path
    dataset_root_dir = r"./datasets/3D-CBCT-Tooth/train"
    images_dir = os.path.join(dataset_root_dir, "images")
    labels_dir = os.path.join(dataset_root_dir, "labels")
    cnt = 0
    # traverse each images
    for image_name in tqdm(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, image_name)
        image = utils.load_image_or_label(image_path, params["resample_spacing"], type="image")
        label = utils.load_image_or_label(label_path, params["resample_spacing"], type="label")
        label[label == 1] = 255
        segmented_images_list = []
        # traverse each models
        for model_name in model_names:
            params["model_name"] = model_name
            params["pretrain"] = os.path.join(r"./pretrain", "Tooth_" + model_name + ".pth")
            # initialize model
            model = models.get_model(params)
            # load model weights
            pretrain_state_dict = torch.load(params["pretrain"], map_location=lambda storage, loc: storage.cuda(params["device"]))
            model_state_dict = model.state_dict()
            for param_name in model_state_dict.keys():
                if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
            model.load_state_dict(model_state_dict, strict=True)
            # segment
            tester = testers.get_tester(params, model)
            segmented_image = tester.test_single_image_without_label(image.copy())
            segmented_images_list.append(segmented_image)
        # traverse all slices of each image
        for slice_ind in range(0, label.shape[2]):
            max_dsc_score = 0
            max_model_name = None
            segment_result_slices_list = []
            for j, segmented_image in enumerate(segmented_images_list):
                segment_result_slice, dsc_score = get_analyse_image_and_dsc(segmented_image[:, :, slice_ind].copy(), label[:, :, slice_ind].copy())
                segment_result_slices_list.append(segment_result_slice)
                if dsc_score > max_dsc_score:
                    max_dsc_score = dsc_score
                    max_model_name = model_names[j]
            if max_model_name == "PMFSNet":
                image_slice = cv2.resize(image[:, :, slice_ind], (224, 224), interpolation=cv2.INTER_AREA)
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice *= 255
                image_slice = image_slice.astype(np.uint8)
                cv2.imwrite(os.path.join(result_dir, "{:04d}".format(cnt) + "_00.jpg"), image_slice)
                label_slice = cv2.resize(label[:, :, slice_ind], (224, 224), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(result_dir, "{:04d}".format(cnt) + "_01.jpg"), label_slice)
                for j, segment_result_slice in enumerate(segment_result_slices_list):
                    segment_result_slice = cv2.resize(segment_result_slice, (224, 224), interpolation=cv2.INTER_NEAREST)
                    segment_result_slice = segment_result_slice[:, :, ::-1]
                    cv2.imwrite(os.path.join(result_dir, "{:04d}".format(cnt) + "_" + "{:02d}".format(j + 2) + ".jpg"), segment_result_slice)


def concat_segmented_image(result_dir, scale=1):
    # create the final image
    image = np.full((742, 2614, 3), 255)
    # traverse the samples
    for i in range(3):
        for j in range(11):
            pos_x, pos_y = i * (224 + 10), j * (224 + 10) + 50
            img = cv2.imread(os.path.join(result_dir, str(i) + "_{:02d}".format(j) + ".jpg"))
            img = cv2.resize(img, (224, 224))
            image[pos_x: pos_x + 224, pos_y: pos_y + 224, :] = img
    image = image[:, :, ::-1]

    # set the text
    col_names = ["Image", "Ground Truth", "U-Net", "AttU-Net", "CA-Net", "CE-Net", "CPF-Net", "CKDNet", "SwinUnet", "DATransUNet", "PMFSNet"]
    row_names = ["a)", "b)", "c)"]
    col_positions = [270, 445, 735, 945, 1190, 1430, 1650, 1885, 2110, 2312, 2585]
    col_positions = [x - 150 for x in col_positions]
    row_positions = [100, 334, 568]
    row_left_positions = [5, 5, 5]

    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    # add text
    for i, text in enumerate(col_names):
        position = (col_positions[i], 697)
        draw.text(position, text, font=font, fill=color)
    for i, text in enumerate(row_names):
        position = (row_left_positions[i], row_positions[i])
        draw.text(position, text, font=font, fill=color, stroke_width=1)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    print(image.size)
    image.save(os.path.join(result_dir, "Kvasir_SEG_Segmentation.jpg"))


if __name__ == "__main__":
    # generate segmented images
    # generate_segment_result_images(
    #     "Kvasir-SEG",
    #     ["UNet", "AttU_Net", "CANet", "CENet", "CPFNet", "CKDNet", "SwinUnet", "DATransUNet", "PMFSNet"]
    # )

    # concatenate the segmented image
    # concat_segmented_image(
    #     r"./images/Kvasir-SEG_segmented_image",
    #     scale=1
    # )

    # generate tooth segmented images
    generate_tooth_segment_result_images(["UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet", "PMFSNet"])
