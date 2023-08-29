import os
import sys
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    recall_score,
)


def get_mimic_meta_data(f_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(os.path.join(f_path))
    df.dropna(subset=['race', 'gender', 'anchor_age'], inplace=True)

    df = df[df['race'].str.contains('ASIAN|BLACK|WHITE')]
    df['race'] = df['race'].apply(categorize_mimic_race)
    df['race'] = pd.Categorical(
        df['race'], categories=['WHITE', 'BLACK', 'ASIAN'], ordered=True
    )

    df = df[df['ViewPosition'].isin(['AP', 'PA'])]

    df['anchor_age'] = df['anchor_age'] / 100

    df = df.rename(columns={'gender': 'sex', 'anchor_age': 'age'})

    df['sex'] = pd.Categorical(df['sex'], categories=['M', 'F'], ordered=True)

    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validate'].copy()
    test_df = df[df['split'] == 'test'].copy()

    train_df['idx'] = range(len(train_df))
    val_df['idx'] = range(len(train_df), len(train_df) + len(val_df))
    test_df['idx'] = range(
        len(train_df) + len(val_df), len(train_df) + len(val_df) + len(test_df)
    )
    return train_df, val_df, test_df


def get_chexpert_meta_data(
    d_path: str, target_file: str = 'chexpert_meta.csv'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    f_path = os.path.join(d_path, target_file)

    if not os.path.exists(f_path):
        preprocess_chexpert_meta(d_path, target_file)

    df = pd.read_csv(f_path)

    df['sex'] = pd.Categorical(df['sex'], categories=['M', 'F'], ordered=True)
    df['race'] = pd.Categorical(
        df['race'], categories=['WHITE', 'BLACK', 'ASIAN'], ordered=True
    )

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validate']
    test_df = df[df['split'] == 'test']

    return train_df, val_df, test_df


def preprocess_chexpert_meta(
    d_path: str, target_file: str = 'chexpert_meta.csv'
) -> None:
    # The full procedure follows chexploration from Glocker et al
    df_demo = pd.DataFrame(pd.read_excel(os.path.join(d_path, 'CHEXPERT DEMO.xlsx')))

    df_demo = df_demo.rename(
        columns={
            'PATIENT': 'patient_id',
            'PRIMARY_RACE': 'race',
            'GENDER': 'sex',
            'AGE_AT_CXR': 'age',
            'ETHNICITY': 'ethnicity',
        }
    )

    # Load split from Gichoya et al und declare index column
    path = os.path.join(d_path, 'chexpert_split_2021_08_20.csv')
    df_data_split = pd.read_csv(path).set_index('index')

    # Inner join on previously obtained indices
    df_img_data = pd.read_csv(os.path.join(d_path, 'train.csv'))
    df_img_data = pd.concat([df_img_data, df_data_split], axis=1)
    df_img_data = df_img_data[~df_img_data['split'].isna()]

    # Isolate patient id
    split = df_img_data.Path.str.split('/', expand=True)
    df_img_data['patient_id'] = split[2]

    # Take protected features from df_demo
    df_img_data.drop(['Age', 'Sex'], inplace=True, axis=1)

    # Combine split with metadata
    df = df_demo.merge(df_img_data, on='patient_id')

    # Rescale age column
    df['age'] = df['age'] / 100

    # Change sex values
    df['sex'] = np.where(df['sex'] == 'Male', 'M', 'F')

    # Filter for race
    df = df[df['race'].str.contains('Asian|Black|White')]
    df['race'] = df['race'].apply(categorize_chexpert_race)

    # Filter for view position
    df = df[df['Frontal/Lateral'] == 'Frontal']

    df['idx'] = range(len(df))

    # Save results to csv
    df.to_csv(os.path.join(d_path, target_file))


def categorize_mimic_race(lab: str) -> str:
    if 'ASIAN' in lab:
        return 'ASIAN'
    elif 'BLACK' in lab:
        return 'BLACK'
    elif 'WHITE' in lab:
        return 'WHITE'
    else:
        return 'OTHER'


def categorize_chexpert_race(lab: str) -> str:
    if 'Asian' in lab:
        return 'ASIAN'
    elif 'Black' in lab:
        return 'BLACK'
    elif 'White' in lab:
        return 'WHITE'
    else:
        return 'OTHER'


def eval_predictions(true: np.ndarray, pred: np.ndarray, do_print: bool = True) -> Dict:
    pred_classes = pred > 0.5

    auc = roc_auc_score(true, pred)
    acc = accuracy_score(true, pred_classes)
    prec, sens, f1, _ = precision_recall_fscore_support(
        true, pred_classes, average='binary'
    )
    spec = recall_score(true, pred_classes, pos_label=0)

    if do_print:
        print(
            'METRICS:\tAUC {:.4f} | ACC {:.4f} | SENS {:.4f} | SPEC {:.4f} | PREC {:.4f} |'
            ' F1 {:.4f}'.format(auc, acc, sens, spec, prec, f1)
        )

    return {
        'AUC': auc,
        'ACC': acc,
        'SENS': sens,
        'SPEC': spec,
        'PREC': prec,
        'F1': f1
    }
