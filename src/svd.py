from __future__ import annotations

import numpy as np
import torch


class SVD:
    def __init__(self) -> None:
        self.u = np.empty(0)
        self.s = np.empty(0)
        self.vh = np.empty(0)

        self.explained_variance_ratio_ = np.empty(0)

    def fit(self, x: np.ndarray) -> SVD:
        # Use torch implementation as it is multiple times faster than numpy svd.
        u, s, vh = torch.linalg.svd(torch.from_numpy(x), full_matrices=False)

        self.u = u.numpy()
        self.s = s.numpy()
        self.vh = vh.numpy()

        self.explained_variance_ratio_ = SVD._compute_explained_variances(self.s)

        return self

    def transform(self, x: np.ndarray, num_components: int) -> np.ndarray:
        return x @ self.vh.T[:, :num_components]

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x @ self.vh[: x.shape[1], :]

    @staticmethod
    def _compute_explained_variances(s: np.ndarray) -> np.ndarray:
        s_squared = np.square(s)
        return np.cumsum(s_squared) / np.sum(s_squared)

    def get_num_components(self, expl_var: float) -> int:
        return np.min(np.where(self.explained_variance_ratio_ > expl_var)) + 1

    def get_total_variance(self, num_components: int) -> float:
        return self.explained_variance_ratio_[num_components - 1]


class PCA(SVD):
    def __init__(self) -> None:
        super().__init__()
        self.mean_ = np.empty(0)

    def fit(self, x: np.ndarray) -> SVD:
        self.mean_ = np.mean(x, axis=0)
        x_center = x - self.mean_
        super().fit(x_center)
        return self

    def transform(self, x: np.ndarray, num_components: int) -> np.ndarray:
        x_center = x - self.mean_
        return super().transform(x_center, num_components)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return super().inverse_transform(x) + self.mean_
