from __future__ import annotations

import os
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    recall_score,
)

DATA_DIR = 'data'
EMBEDDING_FILE = 'embeddings.npy'
META_FILE = 'mimic_meta.csv'
TARGET_DISEASE = 'Pleural Effusion'


class Orthogonalizator:
    def __init__(self) -> None:
        self.q: Optional[np.ndarray] = None
        self.r: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> Orthogonalizator:
        self.q, self.r = np.linalg.qr(x)
        self.beta = np.linalg.multi_dot([np.linalg.inv(self.r), self.q.T, y])
        return self

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(x, y)
        y_hat = np.linalg.multi_dot([self.q, self.q.T, y])
        return y - y_hat

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.q is None:
            raise ValueError('`.fit()` must be called before transformation.')

        return y - x @ self.beta


def categorize_mimic_race(lab: str) -> str:
    if 'ASIAN' in lab:
        return 'ASIAN'
    elif 'BLACK' in lab:
        return 'BLACK'
    elif 'WHITE' in lab:
        return 'WHITE'
    else:
        return 'OTHER'


def eval_predictions(true: np.ndarray, pred: np.ndarray) -> None:
    pred_classes = pred > 0.5

    auc = roc_auc_score(true, pred)
    acc = accuracy_score(true, pred_classes)
    prec, sens, f1, _ = precision_recall_fscore_support(
        true, pred_classes, average='binary'
    )
    spec = recall_score(true, pred_classes, pos_label=0)

    print(
        'METRICS:\tAUC {:.4f} | ACC {:.4f} | SENS {:.4f} | SPEC {:.4f} | PREC {:.4f} |'
        ' F1 {:.4f}'.format(auc, acc, sens, spec, prec, f1)
    )


def get_mimic_meta_data(f_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(os.path.join(f_path))
    df.dropna(subset=['race', 'gender', 'anchor_age'], inplace=True)

    df = df[df['race'].str.contains('ASIAN|BLACK|WHITE')]
    df['race'] = df['race'].apply(categorize_mimic_race)

    df = df[df['ViewPosition'].isin(['AP', 'PA'])]

    df['anchor_age'] = df['anchor_age'] / 100

    df = df.rename(columns={'gender': 'sex', 'anchor_age': 'age'})

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validate']
    test_df = df[df['split'] == 'test']

    return train_df, val_df, test_df


def main() -> None:
    # LOAD DATA
    # ----------------------------------------------------------------------------------
    train_df, val_df, test_df = get_mimic_meta_data(os.path.join(DATA_DIR, META_FILE))
    print(
        f'DATASET SIZES: TRAIN {len(train_df)} | VAL {len(val_df)} | TEST {len(test_df)}'
    )

    emb = np.load(os.path.join(DATA_DIR, EMBEDDING_FILE))
    train_emb = emb[train_df['idx']]
    test_emb = emb[test_df['idx']]

    # DO DIM REDUCTION
    # ----------------------------------------------------------------------------------
    print('Fitting PCA ...')
    pca = PCA(n_components=111)
    train_emb = pca.fit_transform(train_emb)
    test_emb = pca.transform(test_emb)

    # CREATE RESPONSE
    # ----------------------------------------------------------------------------------
    # Follow U-zeros strategy for now
    train_df['response'] = (train_df[TARGET_DISEASE] == 1).astype(int)
    test_df['response'] = (test_df[TARGET_DISEASE] == 1).astype(int)

    # CREATE DESIGN MATRIX
    # ----------------------------------------------------------------------------------
    formula = '1 ~ age + sex + race'
    _, x_train = dmatrices(formula, data=train_df)
    _, x_test = dmatrices(formula, data=test_df)

    # CLEANSE EMBEDDING
    # ----------------------------------------------------------------------------------
    print('Orthogonalizing data ...')
    ortho = Orthogonalizator()
    train_emb_proj = ortho.fit_transform(x_train, train_emb)
    test_emb_proj = ortho.transform(x_test, test_emb)

    # FIT MODEL
    # ----------------------------------------------------------------------------------
    model = LogisticRegression(
        solver='saga',
        max_iter=100,
        verbose=1,
        n_jobs=-1,
    )

    model.fit(train_emb_proj, train_df['response'].tolist())

    train_df['preds'] = model.predict_proba(train_emb_proj)[:, 1]
    test_df['preds'] = model.predict_proba(test_emb_proj)[:, 1]

    print('-' * 75 + '\nTRAINING')
    eval_predictions(train_df['response'], train_df['preds'])
    print('-' * 75 + '\nTESTING')
    eval_predictions(test_df['response'], test_df['preds'])

    # CHECK P-VALUES
    # ----------------------------------------------------------------------------------
    formula = 'preds ~ 1 + age + sex + race'
    mod = sm.OLS.from_formula(formula, data=train_df).fit()
    print(mod.summary())
    print(sm.stats.anova_lm(mod))


if __name__ == '__main__':
    main()
