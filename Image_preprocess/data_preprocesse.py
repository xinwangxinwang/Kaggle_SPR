import os
import pandas as pd
import numpy as np
import re
import pydicom
import pylibjpeg
# import png
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
import warnings
warnings.filterwarnings("ignore")


def imgunit8(img):
    mammogram_dicom = img
    orig_min = mammogram_dicom.min()
    orig_max = mammogram_dicom.max()
    target_min = 0.0
    target_max = 255.0
    mammogram_scaled = (mammogram_dicom - orig_min) * \
                       ((target_max - target_min) / (orig_max - orig_min)) + target_min
    mammogram_uint8_by_function = mammogram_scaled.astype(np.uint8)
    return mammogram_uint8_by_function


def imgunit16(img):
    mammogram_dicom = img
    orig_min = mammogram_dicom.min()
    orig_max = mammogram_dicom.max()
    target_min = 0.0
    target_max = 65535.0
    mammogram_scaled = (mammogram_dicom - orig_min) * \
                       ((target_max - target_min) / (orig_max - orig_min)) + target_min
    mammogram_uint8_by_function = mammogram_scaled.astype(np.uint16)
    return mammogram_uint8_by_function


# function to remove texts in the image
# only breast remained
def segment_breast(img, low_int_threshold=0.05, only_breast_bbox=True):
    # create img for thresholding and contours
    img_8u = (img.astype('float32') / img.max() * 255).astype('uint8')
    if low_int_threshold < 1:
        low_th = int(img_8u.max() * low_int_threshold)
    else:
        low_th = int(low_int_threshold)
    _, img_bin = cv2.threshold(img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [cv2.contourArea(cont) for cont in contours]
    idx = np.argmax(cont_areas)
    breast_mask = cv2.drawContours(np.zeros_like(img_bin), contours, idx, 255, -1)

    # segment the breast
    img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)

    # the case were we want the whole image (with breast segmented and everywhere else 0's) to be returned
    if not only_breast_bbox:
        return img_breast_only, None

    # otherwise return only the bounding box around the segmented breast
    x, y, w, h = cv2.boundingRect(contours[idx])
    img_breast_only = img_breast_only[y:y + h, x:x + w]
    return img_breast_only, (x, y, w, h)


def pad2square(img):
    w = img.shape[1]
    h = img.shape[0]
    min_pixel_value = np.min(img)

    if h > w:
        img_new = np.full((h, h), min_pixel_value)
    else:
        img_new = np.full((w, w), min_pixel_value)

    img_new[int((img_new.shape[0] - img.shape[0]) / 2):int((img_new.shape[0] - img.shape[0]) / 2) + img.shape[0],
    int((img_new.shape[1] - img.shape[1]) / 2):int((img_new.shape[1] - img.shape[1]) / 2) + img.shape[1]] = img

    square_img = img_new
    return square_img


def pad2rectangle(img):
    w = img.shape[1]
    h = img.shape[0]
    min_pixel_value = np.min(img)

    if h > 2 * w:
        img_new = np.full((h, int(h // 2)), min_pixel_value)
    else:
        img_new = np.full((int(w * 2), w), min_pixel_value)

    img_new[int((img_new.shape[0] - img.shape[0]) / 2):int((img_new.shape[0] - img.shape[0]) / 2) + img.shape[0],
    int((img_new.shape[1] - img.shape[1]) / 2):int((img_new.shape[1] - img.shape[1]) / 2) + img.shape[1]] = img

    square_img = img_new
    return square_img


# Get DICOM image metadata
class DCM_Tags():
    def __init__(self, img_dcm):
        try:
            self.laterality = img_dcm.ImageLaterality
        except AttributeError:
            self.laterality = np.nan

        try:
            self.view = img_dcm.ViewPosition
        except AttributeError:
            self.view = np.nan

        try:
            self.orientation = img_dcm.PatientOrientation
        except AttributeError:
            self.orientation = np.nan


# Check whether DICOM should be flipped
def check_dcm(imgdcm):
    # Get DICOM metadata
    tags = DCM_Tags(imgdcm)

    # If image orientation tag is defined
    if ~pd.isnull(tags.orientation):
        # CC view
        if tags.view == 'CC':
            if tags.orientation[0] == 'P':
                flipHorz = True
            else:
                flipHorz = False

            if (tags.laterality == 'L') & (tags.orientation[1] == 'L'):
                flipVert = True
            elif (tags.laterality == 'R') & (tags.orientation[1] == 'R'):
                flipVert = True
            else:
                flipVert = False

        # MLO or ML views
        elif (tags.view == 'MLO') | (tags.view == 'ML'):
            if tags.orientation[0] == 'P':
                flipHorz = True
            else:
                flipHorz = False

            if (tags.laterality == 'L') & ((tags.orientation[1] == 'H') | (tags.orientation[1] == 'HL')):
                flipVert = True
            elif (tags.laterality == 'R') & ((tags.orientation[1] == 'H') | (tags.orientation[1] == 'HR')):
                flipVert = True
            else:
                flipVert = False

        # Unrecognized view
        else:
            flipHorz = False
            flipVert = False

    # If image orientation tag is undefined
    else:
        # Flip RCC, RML, and RMLO images
        if (tags.laterality == 'R') & ((tags.view == 'CC') | (tags.view == 'ML') | (tags.view == 'MLO')):
            flipHorz = True
            flipVert = False
        else:
            flipHorz = False
            flipVert = False

    return flipHorz, flipVert


def save_img(img, base_dir, patient_id, image_id):
    new_img_folder = '{}/{}'.format(base_dir, patient_id)
    os.makedirs(new_img_folder, exist_ok=True)
    new_img_path = '{}/{}'.format(new_img_folder, image_id)
    # cv2.imwrite(new_img_path, image)
    cv2.imwrite(new_img_path, img)
    # print(new_img_path, "finish")


def dicom_file_to_ary(path):
    dicom = pydicom.read_file(path)
    try:
        data = dicom.pixel_array
        if data.min() < data.max():
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                data = np.amax(data) - data
            return data
        else:
            print(path, 'wrong!!!!')
            return None
    except:
        print(path, 'wrong!!!!')
        return None


def read_mammo_from_dcm(dcm_path):
    # Load DICOM
    dcm = pydicom.dcmread(dcm_path, force=True)
    img = dcm.pixel_array

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = np.amax(img) - img

    # # Check if a horizontal flip is necessary
    # horz, _ = check_dcm(dcm)
    # if horz:
    #     # Flip img horizontally
    #     img = np.fliplr(img)

    print(img.shape, img.dtype, img.min(), img.max(), img.mean(), img.std())

    img = imgunit16(img)
    return img


def process_file(args):
    src_path, dest_root, target_image_size = args
    try:
        raw_dir = dest_root / 'raw_png'
        processed_dir = dest_root / 'processed_png'

        # parse the src_path to [base dir /patient id / file name]
        src_path = Path(src_path)
        # base_dir = src_path.parent
        pid = src_path.parent.name
        img_name = src_path.name[:-4] + ".png"
        img = None

        # check if the file is already processed
        if not (raw_dir / pid / img_name).exists():
            img = read_mammo_from_dcm(src_path)
            # convert to dcm to png directly
            save_img(img, raw_dir, pid, img_name)

        if not (processed_dir / pid / img_name).exists():
            if img is None:
                img = read_mammo_from_dcm(src_path)
            img, _ = segment_breast(img, only_breast_bbox=True)
            img = pad2rectangle(img)
            img = cv2.resize(img.astype(np.uint16), target_image_size)
            # convert to dcm to processed png
            save_img(img, processed_dir, pid, img_name)

    except Exception as e:
        print(f"Failed to process {src_path}: {e}")


def get_files(folder):
    for file in folder.rglob('*.dcm'):
    # for file in folder.rglob('*'):
        if "__MACOSX" in str(file):
            continue
        if file.is_file():
            yield file


def main(src_folder, dest_folder, target_image_size, num_workers=None):
    src_folder = Path(src_folder)
    dest_folder = Path(dest_folder)

    if num_workers is None:
        num_workers = cpu_count()

    files = get_files(src_folder)
    processed_files = 0
    print(f"Will import from {src_folder} to {dest_folder}")
    with Pool(num_workers) as pool:
        for _ in pool.imap_unordered(process_file, ((file, dest_folder, target_image_size) for file in files)):
            processed_files += 1
            if processed_files % 1000 == 0:
                print(f"Processed {processed_files} files")

    print('finish all')


def debug():
    src_folder = "/data/groups/public/archive/Kaggle_SPR_Screening_Mammography/dicoms/"
    dest_folder = "/data/groups/public/derived/Kaggle_SPR_Screening_Mammography/pngs/"
    os.makedirs(dest_folder, exist_ok=True)

    src_folder = Path(src_folder)
    dest_folder = Path(dest_folder)

    print(src_folder)
    # src_path = f"{str(src_folder)}/{files_}"
    src_path = src_folder / '002147' / '1.2.840.12345.12321289167504455985471963332987662342247379533335.dcm'
    print(src_path)

    img = read_mammo_from_dcm(src_path)

    pid = src_path.parent.name
    print(pid)
    img_name = src_path.name[:-4] + ".png"
    print(img_name)

    raw_dir = dest_folder / 'raw_png'
    print(raw_dir)
    processed_dir = dest_folder / 'processed_png'
    print(processed_dir)

    print(src_path.exists())

    raw_file = raw_dir / pid / img_name
    print(raw_file)

    process_file = processed_dir / pid / img_name
    print(process_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process DICOM files to PNG.')
    parser.add_argument('--src_folder', type=str, required=True, help='Source folder containing DICOM files')
    parser.add_argument('--dest_folder', type=str, required=True, help='Destination folder for PNG files')
    parser.add_argument('--target_image_size', type=tuple, default=(1024, 2048), help='Target image size for resizing')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of worker processes')
    args = parser.parse_args()
    src_folder = args.src_folder
    dest_folder = args.dest_folder
    target_image_size = args.target_image_size
    num_workers = args.num_workers

    # debug() # for debugging
    # target_image_size = (1024, 2048)
    # src_folder = "/data/groups/public/archive/Kaggle_SPR_Screening_Mammography/dicoms/"
    # dest_folder = "/data/groups/public/derived/Kaggle_SPR_Screening_Mammography/pngs/"
    os.makedirs(dest_folder, exist_ok=True)

    main(src_folder, dest_folder, target_image_size, num_workers)