from __future__ import annotations

from typing import Optional

import numpy as np


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

