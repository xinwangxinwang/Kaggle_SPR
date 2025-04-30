import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

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

# Load the test CSV
test_csv_path = "../data_csv/sample_submissionA.csv"
test_df = pd.read_csv(test_csv_path)
test_df['AccessionNumber'] = test_df['AccessionNumber'].astype(str).str.zfill(6)
print("number of test patient df:", len(test_df['AccessionNumber'].unique()))
test_AccessionNumber_list = test_df['AccessionNumber'].tolist()


# Load and preprocess the training CSV
df = pd.read_csv("../data_csv/train_updated.csv")
df['PatientID'] = df['PatientID'].astype(str).str.zfill(6)
df['AccessionNumber'] = df['AccessionNumber'].astype(str).str.zfill(6)
print("number of training patient df:", len(df['AccessionNumber'].unique()))
# Load and preprocess DICOM metadata
mata_df = pd.read_csv("../data_csv/dcm_tags.csv")
print("number of images in DICOM metadata:", len(mata_df), "number of patients:", len(mata_df['AccessionNumber'].unique()))
# Fill missing or invalid age with default "000Y"
mata_df['PatientAge'] = mata_df['PatientAge'].replace("Not found", pd.NA)
mata_df['PatientAge'] = mata_df['PatientAge'].fillna("000Y")

# Standardize ID formats to 6-digit strings
mata_df['PatientID'] = mata_df['PatientID'].astype(str).str.zfill(6)
mata_df['AccessionNumber'] = mata_df['AccessionNumber'].astype(str).str.zfill(6)
mata_df['SOPInstanceUID'] = mata_df['SOPInstanceUID'].astype(str)

# Add standardized and derived columns
mata_df['patient_id'] = mata_df['PatientID']
mata_df['exam_id'] = mata_df['AccessionNumber']
mata_df['image_id'] = mata_df['SOPInstanceUID'].str.zfill(6)
mata_df['view'] = mata_df['ViewPosition'].astype(str)
mata_df['laterality'] = mata_df['ImageLaterality'].astype(str)

# Extract numeric age from string like "045Y" → 45
mata_df['age'] = mata_df['PatientAge'].str.extract(r'(\d+)').astype(int)

# Initialize label columns
mata_df['density'] = -1
mata_df['label'] = 0

# Create image path: AccessionNumber/SOPInstanceUID.png
mata_df['PATH'] = mata_df['AccessionNumber'] + "/" + mata_df['SOPInstanceUID'] + ".png"
mata_df['dataset'] = 'spr'

# Initialize with default 'N' for missing Laterality labels
mata_df['label_Laterality'] = 'N'

# Use merge instead of a loop to get 'label_Laterality' from training labels
label_df = df[['AccessionNumber', 'Laterality']].copy()
label_df = label_df.rename(columns={'Laterality': 'label_Laterality'})
mata_df = mata_df.merge(label_df, on='AccessionNumber', how='left', suffixes=('', '_merged'))
print("number of images in DICOM metadata:", len(mata_df), "number of patients:", len(mata_df['AccessionNumber'].unique()))


# select only the images that are in the test set
mata_df = mata_df[mata_df['AccessionNumber'].isin(test_AccessionNumber_list)]

# Fill merged Laterality labels if available
mata_df['label_Laterality'] = mata_df['label_Laterality_merged'].fillna(mata_df['label_Laterality'])
mata_df.drop(columns=['label_Laterality_merged'], inplace=True)
print("number of images in DICOM metadata:", len(mata_df), "number of patients:", len(mata_df['AccessionNumber'].unique()))
# Define label logic based on Laterality match
def get_label(row):
    label_laterality = row['label_Laterality']  # Source: df (B, L, R, N)
    image_laterality = row['laterality']        # Source: DICOM (L or R)

    if label_laterality == 'N':
        return 0
    elif label_laterality == 'B':
        return 1
    elif label_laterality == 'L':
        return int(image_laterality == 'L')
    elif label_laterality == 'R':
        return int(image_laterality == 'R')
    return 0

# Apply the label mapping to the DataFrame
mata_df['label'] = mata_df.apply(get_label, axis=1)
print("number of images in DICOM metadata:", len(mata_df), "number of patients:", len(mata_df['AccessionNumber'].unique()))

# Save the updated DataFrame to a new CSV file
output_csv_path = "../data_csv/test_updated_with_label.csv"
mata_df.to_csv(output_csv_path, index=False)
print()