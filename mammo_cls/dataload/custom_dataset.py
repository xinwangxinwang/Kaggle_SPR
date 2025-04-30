import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    """
    Custom dataset for loading mammogram images with associated metadata.

    Args:
        args: Arguments containing image size and other parameters.
        data_info: DataFrame containing image paths and metadata.
        transform: Transformations to be applied to the images.
    """

    def __init__(self, args, data_info, transform):
        self.img_size = (args.img_size, args.img_size // 2)
        self.img_size_cv = (args.img_size // 2, args.img_size)
        # age = data_info['image_age']
        data_info['age'][data_info['age'] == 0] = -1
        # set age range 18-90
        # data_info['age'] = data_info['age'].apply(lambda x: 18 if x < 18 else x)
        # data_info['age'] = data_info['age'].apply(lambda x: 90 if x > 90 else x)
        age = data_info['age']

        self.label = np.asarray(data_info['label'], dtype='int64')
        self.age = np.asarray(age, dtype='float32')
        self.data_info = data_info
        image_file_path = np.asarray(data_info.PATH)
        self.img_file_path = image_file_path
        self.density = np.asarray(data_info['density'], dtype='int64')
        self.birads = np.asarray(data_info['birads'], dtype='int64')
        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at the specified index.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A dictionary containing the image tensor and associated metadata.
        """
        path = self.img_file_path[index]

        age = self.age[index]
        density = self.density[index]
        birads = self.birads[index]
        label = self.label[index]

        pid = self.data_info['patient_id'].iloc[index]
        exam_id = self.data_info['exam_id'].iloc[index]
        img_id = self.data_info['image_id'].iloc[index]

        # Label for the image
        age = np.asarray(age, dtype='float32')
        density = np.asarray(density, dtype='int64')
        birads = np.asarray(birads, dtype='int64')
        label = np.asarray(label, dtype='int64')
        view = self.data_info['view'].iloc[index]
        laterality = self.data_info['laterality'].iloc[index]

        img, img_path = self.__getimg(path)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(image=np.array(img))["image"]

        pid = str(pid)
        exam_id = str(exam_id)
        img_id = str(img_id)

        return {'pid': pid, 'exam_id': exam_id, 'img_id': img_id,
                'view': view, 'laterality': laterality, 'img_path': img_path,
                'img': img, 'age': age, 'density': density, 'birads': birads,'label': label,}

    def __len__(self):
        return len(self.data_info)


    def __getimg(self, path):
        img_path = path
        try:
            image = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            if image is None:
                raise ValueError(f"Image not found or unable to read: {img_path}")
            image = cv2.resize(image.astype(np.uint16), self.img_size_cv)
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
            img = image.astype(np.uint8)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = np.zeros(self.img_size_cv, dtype=np.uint8)  # Return a blank image in case of error
        return img, img_path