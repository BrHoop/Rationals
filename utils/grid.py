from abc import ABC, abstractmethod

import numpy as np


class Grid(ABC):
    """
    """
    def __init__(self, shp, xi, dx, nghost=0):
        """
        Abstract class to define a grid for a PDE system.
        Parameters:
        """
        self.shp = shp
        self.xi = xi
        self.dx = dx
        self.D1 = None
        self.D2 = None
        self.num_filters = 0
        self.Filter = []
        self.nghost = nghost



    def get_shape(self):
        """
        Get the shape of the grid.
        Returns:
        -------
        tuple
            Shape of the grid.
        """
        return self.shp

    def get_nghost(self):
        """
        Get the number of ghost cells.
        Returns:
        -------
        int
            Number of ghost cells.
        """
        return self.nghost

class Grid2D(Grid):
    """
    Class to define a 2D grid for a PDE system.
    Parameters:
    ----------
    Nx : int
        Number of grid points in the x-direction.
    Ny : int
        Number of grid points in the y-direction.
    """

    def __init__(self, params):
        if "Nx" not in params:
            raise ValueError("Nx is required")
        if "Ny" not in params:
            raise ValueError("Ny is required")

        nx = params["Nx"]
        ny = params["Ny"]
        xmin = params.get("Xmin", 0.0)
        xmax = params.get("Xmax", 1.0)
        ymin = params.get("Ymin", 0.0)
        ymax = params.get("Ymax", 1.0)

        dx = (xmax - xmin) / (nx - 1)
        dy = (ymax - ymin) / (ny - 1)

        ng = params.get("NGhost", 0)
        nx = nx + 2 * ng
        ny = ny + 2 * ng
        xmin -= ng * dx
        xmax += ng * dx
        ymin -= ng * dy
        ymax += ng * dy

        shp = [nx, ny]

        xi = [np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)]

        dxn = np.array([dx, dy])
        #print(f"Grid2D: {shp}, {xi}, {dxn}")
        super().__init__(shp, xi, dxn, ng)


