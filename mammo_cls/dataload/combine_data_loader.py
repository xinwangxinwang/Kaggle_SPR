import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def dataloador(myDataset, data_infos, dataset_names, args, train=False):
    """
    Create a DataLoader for the given dataset.

    Args:
        myDataset: Custom dataset class.
        data_infos: List of DataFrames containing image paths and metadata.
        dataset_names: List of dataset names.
        args: Arguments containing image size and other parameters.
        train: Boolean indicating whether to use training transformations.

    Returns:
        DataLoader: DataLoader for the dataset.
    """

    # Define augmentations for training
    augments = [
        A.Resize(args.img_size, args.img_size // 2, interpolation=cv2.INTER_LINEAR),  # Resize images
        A.HorizontalFlip(p=0.5),  # Randomly flip images horizontally
        A.VerticalFlip(p=0.5),  # Randomly flip images vertically
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),  # Random brightness and contrast adjustments
        A.ShiftScaleRotate(rotate_limit=(-5, 5), p=0.3),  # Random shift, scale, and rotation
        A.OneOf([
            A.MedianBlur(blur_limit=3, p=0.3),  # Apply median blur
            A.GaussianBlur(blur_limit=3, p=0.3),  # Apply Gaussian blur
            A.GaussNoise(var_limit=(3.0, 9.0), p=0.3),  # Add Gaussian noise
        ], p=0.3),  # Apply one of the above blurs/noise
        A.OneOf([
            A.ElasticTransform(p=0.3),  # Apply elastic transformation
            A.GridDistortion(num_steps=5, distort_limit=1., p=0.3),  # Apply grid distortion
            A.OpticalDistortion(distort_limit=1., p=0.3),  # Apply optical distortion
        ], p=0.3),  # Apply one of the above distortions
        A.CoarseDropout(max_height=int(args.img_size * 0.1), max_width=int(args.img_size // 2 * 0.1), max_holes=8, p=0.2),  # Apply coarse dropout
        A.Normalize(mean=0.5, std=0.5),  # Normalize images
    ]
    augments.append(ToTensorV2())
    train_transform = A.Compose(augments)

    # Define transformations for testing
    test_transform = A.Compose([
        A.Resize(args.img_size, args.img_size // 2, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])

    # Select transformations based on the training flag
    if train:
        transform, shuffle, drop_last = train_transform, True, True
    else:
        transform, shuffle, drop_last = test_transform, False, False

    num_datasets = len(data_infos)

    # Combine data from multiple datasets
    data_info = []
    for i in range(num_datasets):
        image_dir_ = args.image_dir[i]
        data_info_ = data_infos[i]
        dataset_name = dataset_names[i]
        data_info_ = fit_multi_data_info_format(args, data_info_, dataset_name, image_dir_)
        if args.multi_view:
            raise (NotImplementedError("Multi-view data loading is not implemented yet."))
            # data_info_ = single_view_to_multi_view_data_info(data_info_)
        data_info.append(data_info_)

    # Concatenate all data into a single DataFrame
    data_info = pd.concat(data_info, ignore_index=True)

    # Create the dataset and DataLoader
    dataset = myDataset(args, data_info, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle,
                             num_workers=args.num_workers, pin_memory=True, drop_last=drop_last)

    return data_loader


def datainfo_to_custom(template_datainfo, custom_data_info, file_path, dataset='Inhouse'):
    custom_data_info['PATH'] = custom_data_info['PATH'].apply(lambda x: f"{file_path}/{x}")

    if dataset == 'spr':
        custom_data_info['birads'] = -1

    template_datainfo = custom_data_info[[
        'patient_id', 'exam_id', 'image_id', 'view', 'laterality','age', 'density', 'birads', 'label', 'PATH', 'dataset',
    ]]

    template_datainfo['dataset'] = dataset
    return template_datainfo


def fit_multi_data_info_format(args, custom_data_info, dataset, file_path):
    """
    Organisize custom dataset CSV into some format:
    Example --->
            --->patient_id: 0001 # patient_id
            --->exam_id: 0001 # exam_id
            --->image_id: 0001 # image_id
            --->view: CC # ['CC', 'MLO']
            --->laterality: L # ['L', 'R']
            --->age: 85.5 # age
            --->density: 1 # [-1: density unknown, 0: almost entirely fat, 1: scattered fibroglandular densities, 2: heterogeneously dense, 3: extremely dense]
            --->label: 1 # 0-1
            --->PATH: 'xxx/xxx/xxx.png' # image path
            --->dataset: 'Inhosue'  # ['Inhouse', 'VinDr', 'RSNA']
    """
    template_datainfo = {
        'patient_id':[],
        'exam_id':[],
        'image_id':[],
        'view':[],
        'laterality':[],
        'age':[],
        'density':[],
        'birads':[],
        'label':[],
        'PATH':[],
        'dataset':[],
    }
    datasets = ["inhouse", "embed", "csaw", "cmmd", "vindr", "rsna", "spr"] # TODO: Add more datasets
    if dataset in datasets:
        data_info = datainfo_to_custom(template_datainfo, custom_data_info, file_path, dataset=dataset)
    else:
        raise ValueError(f" DATASET: {args.dataset} is not supported.")

    return data_info


