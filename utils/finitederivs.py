from abc import ABC, abstractmethod

import numpy as np


class SecondDerivative2D(ABC):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    @abstractmethod
    def grad_xx(self, u):
        raise NotImplementedError

    @abstractmethod
    def grad_yy(self, u):
        raise NotImplementedError

class FirstDerivative2D(ABC):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    @abstractmethod
    def grad_x(self, u) -> np.ndarray:
        pass

    @abstractmethod
    def grad_y(self, u) -> np.ndarray:
        pass