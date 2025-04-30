import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold



if __name__ == '__main__':

    CSV_LABEL_PATH = "../data_csv/train_updated_with_label.csv"
    df = pd.read_csv(CSV_LABEL_PATH)

    save_fold_path = f"../data_csv/cv_split"
    os.makedirs(save_fold_path, exist_ok=True)
    spliter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=67)
    df['split_group'] = 'Unknown'
    ret = []
    for i, (train_idxs, val_idxs) in enumerate(spliter.split(df, df.label, groups=df.patient_id)):
        print(f"Fold {i}:")
        df_fold = df.copy()
        df_fold.loc[train_idxs, 'split_group'] = 'train'
        df_fold.loc[val_idxs, 'split_group'] = 'valid'

        test_df = pd.read_csv("../data_csv/test_updated_with_label.csv")
        test_df['split_group'] = 'test'
        df_fold_ = pd.concat([df_fold, test_df], axis=0, ignore_index=True)

        print("train:", len(df_fold_[df_fold_.split_group == 'train']),
              "valid:", len(df_fold_[df_fold_.split_group == 'valid']),
              "test:", len(df_fold_[df_fold_.split_group == 'test']))


        df_fold_.to_csv(f"{save_fold_path}/train_val_fold{i}.csv", index=False)

        # Check the distribution of labels in train and valid sets
        train_label_distribution = df_fold_[df_fold_.split_group == 'train'].label.value_counts(normalize=True)
        valid_label_distribution = df_fold_[df_fold_.split_group == 'valid'].label.value_counts(normalize=True)
        test_label_distribution = df_fold_[df_fold_.split_group == 'test'].label.value_counts(normalize=True)
        print("Train label distribution:\n", train_label_distribution)
        print("Valid label distribution:\n", valid_label_distribution)
        print("Test label distribution:\n", test_label_distribution)
        print('\n--------------------\n\n\n')

    # Done
    print("CV split completed.")