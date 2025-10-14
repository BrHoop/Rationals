from abc import ABC, abstractmethod
import numpy as np
from .grid import Grid
from .types import *


class Equations(ABC):
    """
    Abstract base class for a system of PDEs.
    """

    def __init__(self, NU, g, apply_bc: BCType):
        """
        Initialize the PDE system.

        Parameters:
        NU : int
            The number of PDEs in the system
        grid : Grid
            The spatial grid

        apply_bc : Enum type
            Specifies how boundary conditions are applied, either in the
            RHS routine, or applied to the function after each stage of
            the time integrator.
        """

        self.Nu = NU
        self.shp = g.shp
        self.u = []
        self.apply_bc = apply_bc

        for i in range(NU):
            d = np.zeros(tuple(self.shp))
            self.u.append(d)

    @abstractmethod
    def rhs(self, dtu, u, g):
        """
        The RHS update.
        """
        pass

    @abstractmethod
    def apply_bcs(self, u, g):
        """
        Routine to apply boundary conditions called from time integrator.
        """
        pass

    @abstractmethod
    def initialize(self, g: Grid, params):
        """
        Set the initial data
        """
        pass
