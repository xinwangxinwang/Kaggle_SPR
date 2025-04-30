import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('../')
import warnings
warnings.filterwarnings('ignore')
import time

from mammo_cls.utils.metrics import calculate_classification_metrics as cal_metrics



if __name__ == '__main__':
    # # # ################# ---- ---- #################

    mode_list = ['Mammo-CLIP', 'MIRAI', 'RSNA', 'MIRAI-Aux_Task', 'Mammo-CLIP-Aux_Task']  # 'Mammo-AGE' 'Scratch' 'Mammo-CLIP'

    imgsize_list = [
        # 1024,
        1536,
        2048,
    ]

    arch_list = [
        'efficientnet_b2',
        'efficientnet_b5',
        'resnet18',
        'convnext_small',
    ]

    arch_fold_list = {
        # # New Ensemble top1 model in each fold [1`(2)`, 2`(13)`, 8`(0)`]
        'Mammo-CLIP-efficientnet_b2-1536': [2],
        'Mammo-CLIP-efficientnet_b5-1536': [1],
        'RSNA-convnext_small-2048': [0, 3],
    }

    dataset_name = "SPR"
    results = []
    for mode in mode_list:
        for imgsize in imgsize_list:
            for arch in arch_list:
                if f"{mode}-{arch}-{imgsize}" not in arch_fold_list:
                    continue
                fold_list = arch_fold_list[f"{mode}-{arch}-{imgsize}"]
                for fold in fold_list:
                    dir = f'../logs/finetune_{imgsize}/classification/spr/{mode}-{arch}/{imgsize}_fold{fold}/result_best.csv'
                    print(dir)
                    results.append(pd.read_csv(dir))

    result_df = results[0]
    if len(results) > 1:
        for result_ in results[1:]:
            result_df = result_df.merge(result_, how='inner', on=['patient_id', 'exam_id', 'img_id', 'view', 'laterality', 'risk_label'])

            # # combine the risk scores by averaging the risk score
            result_df['risk_probabilitie'] = (result_df['risk_probabilitie_x'] + result_df['risk_probabilitie_y'])
            # # # combine the risk scores by max the risk score
            # result_df['risk_probabilitie'] = result_df[['risk_probabilitie_x', 'risk_probabilitie_y']].max(axis=1)
            # drop columns: f'{i}_year_risk_x', f'{i}_year_risk_y'
            result_df = result_df.drop(columns=['risk_probabilitie_x', 'risk_probabilitie_y'])
        result_df['risk_probabilitie'] = result_df['risk_probabilitie'] / len(results)

    # get patient level metrics
    # result_df_patient = result_df.groupby(['patient_id', 'exam_id']).agg({'risk_probabilitie': 'mean', 'risk_label': 'max'}).reset_index()
    result_df_breast= result_df.groupby(['patient_id', 'exam_id', 'laterality']).agg({'risk_probabilitie': 'mean', 'risk_label': 'max'}).reset_index()
    result_df_patient= result_df.groupby(['patient_id', 'exam_id']).agg({'risk_probabilitie': 'max', 'risk_label': 'max'}).reset_index()

    metrics = cal_metrics(result_df_patient['risk_probabilitie'], result_df_patient['risk_label'])
    print('Patient level metrics: ')
    print(metrics)

    # generate the submission file
    result_df_patient['AccessionNumber'] = result_df_patient['exam_id'].astype(str).str.zfill(6)
    result_df_patient['target'] = result_df_patient['risk_probabilitie']

    sample_submissionA = result_df_patient[['AccessionNumber', 'target']]
    # generate the submission file with name 'submissionA_date.csv'
    date_ = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    sample_submissionA.to_csv(f'../submissions/submissionA_{date_}.csv', index=False)
    print('submission file generated')