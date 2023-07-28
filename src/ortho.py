from __future__ import annotations

from typing import Optional

import numpy as np


class Orthogonalizator:
    def __init__(self) -> None:
        self.q: Optional[np.ndarray] = None
        self.proj_ortho: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> Orthogonalizator:
        self.q, _ = np.linalg.qr(features)
        self.proj_ortho = self.q.T @ target
        return self

    def fit_transform(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.fit(features, target)
        return target - self.q @ self.proj_ortho

    def transform(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        if self.proj_ortho is None:
            raise ValueError('`.fit()` must be called before transformation.')

        q, _ = np.linalg.qr(features)
        return target - q @ self.proj_ortho
