import os
import cv2
import time
import json
import pydicom
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def get_files(folder):
    for file in folder.rglob('*.dcm'):
    # for file in folder.rglob('*'):
        if "__MACOSX" in str(file):
            continue
        if file.is_file():
            yield file


def get_dicom_tags(args):
    """
    Extracts specific DICOM tags from a DICOM file.

    Args:
        dcm_path (str): Path to the DICOM file.

    Returns:
        dict: A dictionary containing the extracted DICOM tags.
    """
    dcm_path, tags = args
    try:
        # Load the DICOM file
        ds = pydicom.dcmread(dcm_path, force=True)

        # Extract the tags
        return {tag: getattr(ds, tag, None) for tag in tags}

    except Exception as e:
        return {tag: None for tag in tags}


def main(dcm_folder, csv_folder, tags=None, num_workers=None):
    src_folder = Path(dcm_folder)
    csv_folder = Path(csv_folder)

    if num_workers is None:
        num_workers = cpu_count()

    files = get_files(src_folder)

    print(f"Using {num_workers} processes.")
    print(f"Will import from {src_folder} to {csv_folder}")

    results = []
    with Pool(num_workers) as pool:
        for tag_dict in tqdm(pool.imap_unordered(get_dicom_tags, ((file, tags) for file in files)), desc="Processing"):
            results.append(tag_dict)

    df = pd.DataFrame(results)
    output_path = csv_folder / "dcm_tags.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved collected tags to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process DICOM files to PNG.')
    parser.add_argument('--src_folder', type=str, required=True, help='Source folder containing DICOM files')
    parser.add_argument('--dest_folder', type=str, required=True, help='Destination folder for PNG files')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of worker processes')
    args = parser.parse_args()
    src_folder = args.src_folder
    dest_folder = args.dest_folder
    num_workers = args.num_workers

    # src_folder = "/data/groups/public/archive/Kaggle_SPR_Screening_Mammography/dicoms/"
    # dest_folder = "/projects/xin-275d/challenges/Kaggle_SPR/data_csv/"
    start_time = time.time()

    tags = ['PatientID', 'AccessionNumber', 'AcquisitionDate', 'PatientAge', 'PatientBirthDate',
           'SOPInstanceUID', 'SeriesNumber', 'StudyDescription',
           'ImageLaterality', 'SeriesDescription', 'ViewPosition',
           'ImagerPixelSpacing', 'Manufacturer', 'ManufacturerModelName',
           'PhotometricInterpretation', 'PixelSpacing', 'PartialView', 'PartialViewDescription']

    main(src_folder, dest_folder, tags, num_workers)

    elapsed = time.time() - start_time
    print(f"Done! Total time: {elapsed:.2f} seconds.")