import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
"""
patient_id:             ID string for this patient. Is used to link together mammograms for one patient.
exam_id:                ID string for this mammogram. Is used to link together several files for one mammogram. 
                        Note, this code-base assumes that "patient_id + exam_id" is the unique key for a mammogram.
exam_date:
age:                    age of the patient
birads:                 BI-RADS score of the mammogram
density:                Breast density of the breast
race:                   Race of the patient

...
split_group:            Can take values , or to note the training, validation and testing samples.train dev test
"""
# -----------------------------------------------------------------------
template_datainfo = {
    'patient_id': [],
    'exam_id': [],
    'image_id':[],
    'exam_date': [],
    'age': [],
    'birads': [],
    'density': [],
    'view': [],
    'laterality': [],
    'race': [],
    'label':[],
    'PATH':[],
    'dataset':[],
    'split_group': []}


# # ###################  EMBED ##########################
def get_label(row):
    label = -1
    side = 'N'

    laterality = row['laterality']

    # Cnacer diagnosis in 60 days
    if row['rad_recall'] == 1: # canccer diagnosis in 60 days
        if row['x_cancer_laterality'] == 'Left':
            side = 'L'

        elif row['x_cancer_laterality'] == 'Right':
            side = 'R'

        if laterality == side:
            label = 1
            return label


    # Recall
    if row['rad_recall'] == 1:
        if row['rad_recall_type_right'] in [1, 2]:
            side = 'R'

            if laterality == side:
                label = 1
                return label

        if row['rad_recall_type_left'] == [1, 2]:
            side = 'L'

            if laterality == side:
                label = 1
                return label

    return 0


_csv_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/proj-ul-bcr/codes/demo/CSV/embed_all_followup0.csv'
clinical_csv_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-aging/proj-breast-aging/codes/EMBED_Open_Data-main/tables/EMBED_OpenData_clinical_reduced.csv'
_image_dir = "/processing/x.wang/embed_dataset/preprocessed-img-2048/"

clinical_df = pd.read_csv(clinical_csv_dir, header=0)
clinical_df = clinical_df[['empi_anon', 'acc_anon', 'desc', 'side', 'asses', 'numfind']]




birads_dict = {
        'A': 0,
        'N': 1,
        'B': 2,
        'P': 3,
        'S': 4,
        'M': 5,
        'K': 6,
        'X': -1,
    }
clinical_df['birads'] = clinical_df['asses'].map(lambda x: birads_dict[x])
clinical_df['side'][clinical_df['side'].isna()] = 'N'
clinical_df['patient_id'] = clinical_df['empi_anon'].astype(str)
clinical_df['exam_id'] = clinical_df['acc_anon'].astype(str)
clinical_df['patient_id-exam_id'] = clinical_df['patient_id'] + '-' + clinical_df['exam_id']
numfind = len(clinical_df['numfind'].unique())




# generate the dict for the patient_id-exam_id - birads and side
dict_p2birads = {}
for i in range(len(clinical_df)):
    patient_id_exam_id = clinical_df['patient_id-exam_id'].iloc[i]
    birads = clinical_df['birads'].iloc[i]
    side = clinical_df['side'].iloc[i]
    if patient_id_exam_id not in dict_p2birads:
        dict_p2birads[patient_id_exam_id] = (birads, side)
    else:
        old_birads, old_side = dict_p2birads[patient_id_exam_id]
        if old_birads in [1, 2]:
            if birads not in [1, 2]:
                dict_p2birads[patient_id_exam_id] = (birads, side)
            else:
                dict_p2birads[patient_id_exam_id] = (old_birads, old_side)
        else:
            if birads in [1, 2]:
                dict_p2birads[patient_id_exam_id] = (birads, side)
            else:
                if old_side != side:
                    side = "B"
                dict_p2birads[patient_id_exam_id] = (max(birads, old_birads), side)

print(dict_p2birads)


data_info = pd.read_csv(_csv_dir, header=0)
print(len(clinical_df), len(clinical_df.exam_id.unique()))
print(len(data_info), len(data_info.exam_id.unique()))
dataset = 'embed'
data_info['patient_id'] = data_info['patient_id'].astype(str)
data_info['exam_id'] = data_info['exam_id'].astype(str)
data_info['patient_id-exam_id'] = data_info['patient_id'] + '-' + data_info['exam_id']
data_info['image_id'] = data_info['exam_id'].astype(str)

def get_label(x):
    birads, birads_side, laterality = x
    if birads not in [1, 2]:
        if laterality == birads_side or birads_side == 'B':
            label = birads
        else:
            label = 2
    else:
        label = birads
    return label

data_info['age'] = 0
data_info['old_birads'] = data_info['birads']
# update the birads and side of data_info from clinical_df
data_info['birads'] = data_info['patient_id-exam_id'].map(lambda x: dict_p2birads[x][0])
data_info['birads_side'] = data_info['patient_id-exam_id'].map(lambda x: dict_p2birads[x][1])
data_info['birads'] = data_info[['birads', 'birads_side', 'laterality']].apply(lambda x: get_label(x), axis=1)
data_info['label'] = data_info['birads'].apply(lambda x: 0 if x in [1, 2] else 1)
data_info['density'][data_info['density'].isna()] = -1
data_info['density'] = data_info['density'].map({-1: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: -1})
data_info['PATH'] = data_info['file_path'].astype(str).apply(lambda x:str(x).replace(_image_dir, ''))
data_info['split_group'] = 'train'
data_info['dataset'] = dataset
data_info = data_info[data_info['label'] != -1]
print(len(data_info))

new_base_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/challenges/Kaggle_SPR/data_csv/external_datasets'
os.makedirs(new_base_dir, exist_ok=True)
data_info.to_csv(os.path.join(new_base_dir, f'{dataset}_data_info.csv'), index=False)
print(len(data_info))

# # # ###################  CSAW ##########################
# def get_label(row):
#     label = -1
#     side = 'N'
#
#     laterality = row['laterality']
#
#     # Cnacer diagnosis in 60 days
#     if row['rad_recall'] == 1: # canccer diagnosis in 60 days
#         if row['x_cancer_laterality'] == 'Left':
#             side = 'L'
#
#         elif row['x_cancer_laterality'] == 'Right':
#             side = 'R'
#
#         if laterality == side:
#             label = 1
#             return label
#
#
#     # Recall
#     if row['rad_recall'] == 1:
#         if row['rad_recall_type_right'] in [1, 2]:
#             side = 'R'
#
#             if laterality == side:
#                 label = 1
#                 return label
#
#         if row['rad_recall_type_left'] == [1, 2]:
#             side = 'L'
#
#             if laterality == side:
#                 label = 1
#                 return label
#
#     return 0
#
#
# _csv_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-mtpbcr/proj-mtp-bcr/codes/CSAW-CC/csv/CSAW-CC_breast_cancer_screening_data.csv'
# _image_dir = "/projects/mammogram_data/vindr-mammo/raw_png_old"
#
# data_info = pd.read_csv(_csv_dir, header=0)
# dataset = 'csaw'
# data_info['patient_id'] = data_info['anon_patientid']
# data_info['exam_id'] = data_info['exam_year'].astype(str)
# data_info['image_id'] = data_info['anon_filename'].astype(str)
# data_info['age'] = 0
#
# data_info['view'] = data_info['viewposition'].astype(str)
# data_info['laterality'] = data_info['imagelaterality'].astype(str)
# data_info['laterality'] = data_info['laterality'].replace({'Left': 'L', 'Right': 'R'})
#
# data_info['birads'] = -1
#
# data_info['rad_timing'][data_info['rad_timing'].isna()] = -1
# data_info['rad_r1'][data_info['rad_r1'].isna()] = -1
# data_info['rad_r2'][data_info['rad_r2'].isna()] = -1
# data_info['rad_recall'][data_info['rad_recall'].isna()] = -1
# data_info['rad_recall_type_right'][data_info['rad_recall_type_right'].isna()] = -1
# data_info['rad_recall_type_left'][data_info['rad_recall_type_left'].isna()] = -1
#
# data_info['label'] = data_info.apply(lambda row: get_label(row), axis=1)
#
# # 0: 0-25, 1: 26-50, 2: 51-75, 3: 76-100
# data_info['density'] = data_info['libra_percentdensity'].apply(lambda x: 0 if x <= 25 else (1 if x <= 50 else (2 if x <= 75 else 3)))
#
#
# data_info['PATH'] = data_info['image_id'].astype(str).apply(lambda x:str(x).replace('.dcm', '.png'))
# data_info['split_group'] = 'train'
# data_info['dataset'] = dataset
# data_info = data_info[data_info['label'] != -1]
# print(len(data_info))
#
# new_base_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/challenges/Kaggle_SPR/data_csv/external_datasets'
# os.makedirs(new_base_dir, exist_ok=True)
# data_info.to_csv(os.path.join(new_base_dir, f'{dataset}_data_info.csv'), index=False)
# print(len(data_info))


# # # ###################  RSNA ##########################
# def get_label(row):
#     label = -1
#     if row['birads'] != -1:
#         if row['birads'] in [1, 2]:
#             label = 0
#         elif row['birads'] == [0]:
#             label = 0
#
#     if row['cancer'] == 1:
#         label = 1
#
#     if row['biopsy'] == 1:
#         label = 1
#
#     if row['invasive'] == 1:
#         label = 1
#
#     if row['difficult_negative_case'] == 'True':
#         label = 1
#
#     return label
#
# _csv_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/challenges/Kaggle_SPR/data_csv/external_datasets/RSNA-2022-Mammo/train.csv'
# _image_dir = "/projects/mammogram_data/vindr-mammo/raw_png_old"
#
# data_info = pd.read_csv(_csv_dir, header=0)
# dataset = 'rsna'
# data_info['patient_id'] = data_info['patient_id'].astype(str)
# data_info['exam_id'] = data_info['patient_id'].astype(str)
# data_info['image_id'] = data_info['image_id'].astype(str)
# data_info['view'] = data_info['view'].astype(str)
# data_info['laterality'] = data_info['laterality'].astype(str)
# data_info['age'][data_info['age'].isna()] = 0
# data_info['age'] = data_info['age'].astype(float)
#
# data_info['difficult_negative_case'] = data_info['difficult_negative_case'].astype(str)
#
# data_info['birads'] = data_info['BIRADS']
# data_info['birads'][data_info['birads'].isna()] = -1
# data_info['label'] = data_info.apply(lambda row: get_label(row), axis=1)
#
# data_info['density'][data_info['density'].isna()] = 'Unknown'
# data_info['density'] = data_info['density'].map({'Unknown': -1, 'A': 0, 'B': 1, 'C': 2, 'D': 3})
#
# data_info['PATH'] = data_info['patient_id'].astype(str) + '/' + data_info['image_id'].astype(str) + '.png'
# data_info['split_group'] = 'train'
# data_info['dataset'] = dataset
# data_info = data_info[data_info['label'] != -1]
# print(len(data_info))
#
# new_base_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/challenges/Kaggle_SPR/data_csv/external_datasets'
# os.makedirs(new_base_dir, exist_ok=True)
# data_info.to_csv(os.path.join(new_base_dir, f'{dataset}_data_info.csv'), index=False)
# print(len(data_info))

# # # ###################  CMMD ##########################
# _csv_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-aging/proj-breast-aging/codes/Mammo-AGE-Downstream/script/df_preprocess/risk/CMMD_clinicaldata_revision.csv'
# _image_dir = "/projects/mammogram_data/CMMD-data/preprocessed_CMMD_20230506"
#
# data_info = pd.read_csv(_csv_dir, header=0)
# dataset = 'train'
# data_info['patient_id'] = data_info['ID1'].astype(str)
# data_info['exam_id'] = data_info['ID1'].astype(str)
# data_info['race'] = 'asian'
# data_info['birads'] = -1
# data_info['density'] = -1
#
# data_info['years_to_last_followup'] = 100
# data_info['years_to_last_followup'] = data_info['years_to_last_followup'].astype(int)
# data_info['birads'] = data_info['birads'].astype(int)
# data_info['density'] = data_info['density'].astype(int)
# # data_info['age'] = data_info['age'].apply(lambda x: int(str(x).replace("Y", "")))
# data_info['age'] = data_info['Age'].astype(float)
# data_info['exam_date'] = "2020/01/01"
#
# data_info['cancer'] = data_info['classification'].apply(lambda x: 1 if x >= "Malignant" else 0)
#
# for i in range(len(data_info)):
#     image_laterality = data_info['LeftRight'].iloc[i]
#     patient_id = data_info['patient_id'].iloc[i]
#     exam_id = data_info['exam_id'].iloc[i]
#     exam_date = data_info['exam_date'].iloc[i]
#     age = data_info['age'].iloc[i]
#     birads = data_info['birads'].iloc[i]
#     density = data_info['density'].iloc[i]
#     years_to_last_followup = data_info['years_to_last_followup'].iloc[i]
#     race = data_info['race'].iloc[i]
#     malignancy = data_info['cancer'].iloc[i]
#     years_to_cancer = 100 if malignancy == 0 else 0
#
#     template_datainfo['patient_id'].append(patient_id)
#     template_datainfo['exam_id'].append(f"{exam_id}_{image_laterality}")
#     template_datainfo['exam_date'].append(exam_date)
#     template_datainfo['age'].append(age)
#     template_datainfo['birads'].append(birads)
#     template_datainfo['density'].append(density)
#     template_datainfo['years_to_last_followup'].append(years_to_last_followup)
#     template_datainfo['race'].append(race)
#     template_datainfo['malignancy'].append(malignancy)
#     template_datainfo['years_to_cancer'].append(years_to_cancer)
#     template_datainfo['split_group'].append('test')
#
#     image_laterality = data_info['LeftRight'].iloc[i]
#     lateralitys = ['L', 'R']
#     views = ['CC', 'MLO']
#     for laterality in lateralitys:
#         for view in views:
#             side_cancer = 1 if malignancy == 1 and image_laterality == laterality else 0
#             if view == 'CC':
#                 template_datainfo[f'years_to_cancer_{laterality.lower()}'].append(0 if side_cancer == 1 else 100)
#                 template_datainfo[f'malignancy_{laterality.lower()}'].append(side_cancer)
#             image_path = f"{patient_id}_{view}_{image_laterality}.png"
#             template_datainfo[f'PATH_{laterality}_{view}'].append(image_path)
#
# new_base_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/challenges/Kaggle_SPR/data_csv/external_datasets'
# os.makedirs(new_base_dir, exist_ok=True)
# data_info.to_csv(os.path.join(new_base_dir, 'CMMD_data_info.csv'), index=False)


# # # ###################  VINDR ##########################
# _csv_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-aging/proj-breast-aging/codes/breast_age/CSV/VinDr-CSV/all_data_info.csv'
# _image_dir = "/projects/mammogram_data/vindr-mammo/raw_png_old"
#
# data_info = pd.read_csv(_csv_dir, header=0)
# dataset = 'vindr'
# data_info['patient_id'] = data_info['study_id'].astype(str)
# data_info['exam_id'] = data_info['study_id'].astype(str)
# data_info['image_id'] = data_info['image_id'].astype(str)
# data_info['view'] = data_info['view_position'].astype(str)
# data_info['laterality'] = data_info['laterality'].astype(str)
#
# data_info['race'] = 'asian'
# data_info['birads'] = data_info['breast_birads'].map(
#     {'BI-RADS 1': 1, 'BI-RADS 2': 2, 'BI-RADS 3': 3, 'BI-RADS 4': 4, 'BI-RADS 5': 5, })
# data_info['density'] = data_info['breast_density'].map({'DENSITY A': 0, 'DENSITY B': 1, 'DENSITY C': 2, 'DENSITY D': 3})
#
# data_info['years_to_last_followup'] = 100
# data_info['years_to_last_followup'] = data_info['years_to_last_followup'].astype(int)
# data_info['birads'] = data_info['birads'].astype(int)
# data_info['density'] = data_info['density'].astype(int)
# data_info['age'] = data_info['age'].apply(lambda x: int(str(x).replace("Y", "")))
# data_info['age'] = data_info['age'].astype(float)
# # data_info['exam_date'] = "2020/01/01"
# data_info['label'] = data_info['birads'].apply(lambda x: 1 if x >= 3 else 0)
# data_info['cancer'] = data_info['birads'].apply(lambda x: 1 if x >= 5 else 0)
# data_info['PATH'] = data_info['study_id'].astype(str) + '/' + data_info['image_id'].astype(str) + '.png'
# data_info['split_group'] = 'train'
# data_info['dataset'] = dataset
#
# print(len(data_info))
#
# new_base_dir = '/Users/x.wang/PycharmProjects/new_nki_project/xin-275d/challenges/Kaggle_SPR/data_csv/external_datasets'
# os.makedirs(new_base_dir, exist_ok=True)
# data_info.to_csv(os.path.join(new_base_dir, 'vindr_data_info.csv'), index=False)
#
# # # ###################  VINDR ##########################
print(data_info.head())