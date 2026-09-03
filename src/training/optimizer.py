from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Adam:
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: np.ndarray | None = None
    v: np.ndarray | None = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return theta - update, update


@dataclass
class SGD:
    lr: float = 0.01

    def step(self, theta: np.ndarray, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        update = self.lr * grad
        return theta - update, update


def make_optimizer(name: str, lr: float):
    if name.lower() == "adam":
        return Adam(lr=lr)
    if name.lower() == "sgd":
        return SGD(lr=lr)
    raise ValueError(f"Unsupported optimizer {name!r}")

