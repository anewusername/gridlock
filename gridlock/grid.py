from typing import ClassVar, Self
from collections.abc import Callable, Sequence

import numpy
from numpy.typing import NDArray, ArrayLike

import pickle
import warnings
import copy

from . import GridError
from .draw import GridDrawMixin
from .read import GridReadMixin
from .position import GridPosMixin


foreground_callable_type = Callable[[NDArray, NDArray, NDArray], NDArray]


class Grid(GridDrawMixin, GridReadMixin, GridPosMixin):
    """
    Simulation grid metadata for finite-difference simulations.

      Can be used to generate non-uniform rectangular grids (the entire grid
    is generated based on the coordinates of the boundary points). Also does
    straightforward natural <-> grid unit conversion.

      This class handles data describing the grid, and should be paired with a
    (separate) ndarray that contains the actual data in each cell. The `allocate()`
    method can be used to create this ndarray.

    The resulting `cell_data[i, a, b, c]` should correspond to the value in the
    `i`-th grid, in the cell centered around
    ```
          (xyz[0][a] + dxyz[0][a] * shifts[i, 0],
           xyz[1][b] + dxyz[1][b] * shifts[i, 1],
           xyz[2][c] + dxyz[2][c] * shifts[i, 2]).
    ```
     You can get raw edge coordinates (`exyz`),
                   center coordinates (`xyz`),
                           cell sizes (`dxyz`),
      from the properties named as above, or get them for a given grid by using the
      `self.shifted_*xyz(which_shifts)` functions.

     The sizes of adjacent cells are taken into account when applying shifts. The
      total shift for each edge is chosen using `(shift * dx_of_cell_being_moved_through)`.

     It is tricky to determine the size of the right-most cell after shifting,
      since its right boundary should shift by `shifts[i][a] * dxyz[a][dxyz[a].size]`,
      where the dxyz element refers to a cell that does not exist.
     Because of this, we either assume this 'ghost' cell is the same size as the last
      real cell, or, if `self.periodic[a]` is set to `True`, the same size as the first cell.
    """
    exyz: list[NDArray]
    """Cell edges. Monotonically increasing without duplicates."""

    periodic: list[bool]
    """For each axis, determines how far the rightmost boundary gets shifted. """

    shifts: NDArray
    """Offsets `[[x0, y0, z0], [x1, y1, z1], ...]` for grid `0,1,...`"""

    Yee_Shifts_E: ClassVar[NDArray] = 0.5 * numpy.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        ], dtype=float)
    """Default shifts for Yee grid E-field"""

    Yee_Shifts_H: ClassVar[NDArray] = 0.5 * numpy.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        ], dtype=float)
    """Default shifts for Yee grid H-field"""

    def __init__(
            self,
            pixel_edge_coordinates: Sequence[ArrayLike],
            shifts: ArrayLike = Yee_Shifts_E,
            periodic: bool | Sequence[bool] = False,
            ) -> None:
        """
        Args:
            pixel_edge_coordinates: 3-element list of (ndarrays or lists) specifying the
                coordinates of the pixel edges in each dimensions
                (ie, `[[x0, x1, x2,...], [y0,...], [z0,...]]` where the first pixel has x-edges x=`x0` and
                x=`x1`, the second has edges x=`x1` and x=`x2`, etc.)
            shifts: Nx3 array containing `[x, y, z]` offsets for each of N grids.
                E-field Yee shifts are used by default.
            periodic: Specifies how the sizes of edge cells are calculated; see main class
                documentation. List of 3 bool, or a single bool that gets broadcast. Default `False`.

        Raises:
            `GridError` on invalid input
        """
        edge_arrs = [numpy.array(cc) for cc in pixel_edge_coordinates]
        self.exyz = [numpy.unique(edges) for edges in edge_arrs]
        self.shifts = numpy.array(shifts, dtype=float)

        for i in range(3):
            if self.exyz[i].size != edge_arrs[i].size:
                warnings.warn(f'Dimension {i} had duplicate edge coordinates', stacklevel=2)

        if isinstance(periodic, bool):
            self.periodic = [periodic] * 3
        else:
            self.periodic = list(periodic)

        if len(self.shifts.shape) != 2:
            raise GridError('Misshapen shifts: shifts must have two axes! '
                            f' The given shifts has shape {self.shifts.shape}')
        if self.shifts.shape[1] != 3:
            raise GridError('Misshapen shifts; second axis size should be 3,'
                            f' shape is {self.shifts.shape}')

        if (numpy.abs(self.shifts) > 1).any():
            raise GridError('Only shifts in the range [-1, 1] are currently supported')

        if (self.shifts < 0).any():
            # TODO: Test negative shifts
            warnings.warn('Negative shifts are still experimental and mostly untested, be careful!', stacklevel=2)

    @staticmethod
    def load(filename: str) -> 'Grid':
        """
        Load a grid from a file

        Args:
            filename: Filename to load from.
        """
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)

        g = Grid([[-1, 1]] * 3)
        g.__dict__.update(tmp_dict)
        return g

    def save(self, filename: str) -> Self:
        """
        Save to file.

        Args:
            filename: Filename to save to.

        Returns:
            self
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=2)
        return self

    def copy(self) -> Self:
        """
        Returns:
            Deep copy of the grid.
        """
        return copy.deepcopy(self)
