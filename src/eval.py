from enum import Enum
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
from patsy import dmatrices
import statsmodels.api as sm

from src.classifier import get_classifier
from src.svd import PCA
from src.ortho import Orthogonalizator
from src.utils import eval_predictions


class Disease(Enum):
    ENLARGED_CARDIOMEDIASTINUM = 'Enlarged Cardiomediastinum'
    CARDIOMEGALY = 'Cardiomegaly'
    LUNG_OPACITY = 'Lung Opacity'
    LUNG_LESION = 'Lung Lesion'
    EDEMA = 'Edema'
    CONSOLIDATION = 'Consolidation'
    PNEUMONIA = 'Pneumonia'
    ATELECTASIS = 'Atelectasis'
    PNEUMOTHORAX = 'Pneumothorax'
    PLEURAL_EFFUSION = 'Pleural Effusion'
    PLEURAL_OTHER = 'Pleural Other'
    FRACTURE = 'Fracture'
    SUPPORT_DEVICES = 'Support Devices'
    NO_FINDING = 'No Finding'


class EmbeddingEvaluator:
    def __init__(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        train_emb: np.ndarray,
        test_emb: np.ndarray,
        n_components: Optional[int] = None,
    ) -> None:
        # Original dataframes
        self.train_df = train_df
        self.test_df = test_df

        # Protected feature design matrices
        self.train_x, self.test_x = self._create_design_matrices()

        # Do dim reduction if desired
        if n_components is None:
            self.train_emb = train_emb
            self.test_emb = test_emb

        else:
            print('Applying dimensionality reduction.')
            pca = PCA()
            pca.fit(train_emb)
            print(f'Explained variance: {pca.get_total_variance(n_components):.3f}')

            self.train_emb = pca.transform(train_emb, num_components=n_components)
            self.test_emb = pca.transform(test_emb, num_components=n_components)

        # Conduct orthogonalization
        ortho = Orthogonalizator()
        self.train_emb_ortho = ortho.fit_transform(self.train_x, train_emb)
        self.test_emb_ortho = ortho.transform(self.test_x, test_emb)

    def _create_design_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        formula = '1 ~ age + sex + race'
        _, train_x = dmatrices(formula, data=self.train_df)
        _, test_x = dmatrices(formula, data=self.test_df)

        return train_x, test_x

    def eval_classifier(
            self,
            response: Disease = Disease.PLEURAL_EFFUSION,
            ortho: bool = False,
            clf_name: str = 'nn', clf_args: Optional[Dict] = None,
            formula: str = 'scores ~ 1 + age + sex + race'
    ) -> None:
        # Follow U-zeros strategy for now
        self.train_df['response'] = (self.train_df[response.value] == 1).astype(int)
        self.test_df['response'] = (self.test_df[response.value] == 1).astype(int)

        model = get_classifier(clf_name, clf_args)

        # Choose which embedding is the target
        train_emb = self.train_emb_ortho if ortho else self.train_emb
        test_emb = self.test_emb_ortho if ortho else self.test_emb

        model.fit(train_emb, self.train_df['response'].tolist())

        self.train_df['preds'] = model.predict_proba(train_emb)[:, 1]
        self.test_df['preds'] = model.predict_proba(test_emb)[:, 1]

        #print(sum(self.test_df['preds'] > 0.5) / len(self.test_df['preds'] ))

        self.train_df['scores'] = model.decision_function(train_emb)
        self.test_df['scores'] = model.decision_function(test_emb)

        print('-' * 75 + '\nTRAINING')
        eval_predictions(self.train_df['response'], self.train_df['preds'])
        print('-' * 75 + '\nTESTING')
        eval_predictions(self.test_df['response'], self.test_df['preds'])
        print('-' * 75)

        mod = sm.OLS.from_formula(formula, data=self.train_df).fit()
        print(mod.summary())

        print('-' * 75 + '\nANOVA')

        print(sm.stats.anova_lm(mod))

    def get_classifier_metrics(
            self,
            response: Disease = Disease.PLEURAL_EFFUSION,
            clf_name: str = 'nn', clf_args: Optional[Dict] = None,
    ) -> str:
        self.train_df['response'] = (self.train_df[response.value] == 1).astype(int)
        self.test_df['response'] = (self.test_df[response.value] == 1).astype(int)

        # Choose which embedding is the target
        train_emb = self.train_emb
        test_emb = self.test_emb

        model = get_classifier(clf_name, clf_args)
        model.fit(train_emb, self.train_df['response'].tolist())

        self.test_df['preds'] = model.predict_proba(test_emb)[:, 1]
        m = eval_predictions(self.test_df['response'], self.test_df['preds'], do_print=False)

        auc_normal = m['AUC']

        # Choose which embedding is the target
        train_emb = self.train_emb_ortho
        test_emb = self.test_emb_ortho

        model = get_classifier(clf_name, clf_args)
        model.fit(train_emb, self.train_df['response'].tolist())

        self.test_df['preds'] = model.predict_proba(test_emb)[:, 1]
        m = eval_predictions(self.test_df['response'], self.test_df['preds'], do_print=False)

        auc_ortho = m['AUC']

        msg = '{:.3f} & {:.3f}'.format(auc_normal, auc_ortho)
        print(msg)
        return msg


