from __future__ import annotations

from typing import Optional

import numpy as np


class Orthogonalizator:
    def __init__(self) -> None:
        self.q: Optional[np.ndarray] = None
        self.r: Optional[np.ndarray] = None

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.q, self.r = np.linalg.qr(x)
        y_hat = np.linalg.multi_dot([self.q, self.q.T, y])
        return y - y_hat
