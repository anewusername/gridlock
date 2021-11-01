from typing import List, Tuple, Callable, Dict, Optional, Union, Sequence, ClassVar, TypeVar

import numpy    # type: ignore
from numpy import diff, floor, ceil, zeros, hstack, newaxis

import pickle
import warnings
import copy

from . import GridError


foreground_callable_type = Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray], numpy.ndarray]
T = TypeVar('T', bound='Grid')


class Grid:
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
    exyz: List[numpy.ndarray]
    """Cell edges. Monotonically increasing without duplicates."""

    periodic: List[bool]
    """For each axis, determines how far the rightmost boundary gets shifted. """

    shifts: numpy.ndarray
    """Offsets `[[x0, y0, z0], [x1, y1, z1], ...]` for grid `0,1,...`"""

    Yee_Shifts_E: ClassVar[numpy.ndarray] = 0.5 * numpy.array([[1, 0, 0],
                                                               [0, 1, 0],
                                                               [0, 0, 1]], dtype=float)
    """Default shifts for Yee grid E-field"""

    Yee_Shifts_H: ClassVar[numpy.ndarray] = 0.5 * numpy.array([[0, 1, 1],
                                                               [1, 0, 1],
                                                               [1, 1, 0]], dtype=float)
    """Default shifts for Yee grid H-field"""

    from .draw import (
        draw_polygons, draw_polygon, draw_slab, draw_cuboid,
        draw_cylinder, draw_extrude_rectangle,
        )
    from .read import get_slice, visualize_slice, visualize_isosurface
    from .position import ind2pos, pos2ind

    @property
    def dxyz(self) -> List[numpy.ndarray]:
        """
        Cell sizes for each axis, no shifts applied

        Returns:
            List of 3 ndarrays of cell sizes
        """
        return [numpy.diff(ee) for ee in self.exyz]

    @property
    def xyz(self) -> List[numpy.ndarray]:
        """
        Cell centers for each axis, no shifts applied

        Returns:
            List of 3 ndarrays of cell edges
        """
        return [self.exyz[a][:-1] + self.dxyz[a] / 2.0 for a in range(3)]

    @property
    def shape(self) -> numpy.ndarray:
        """
        The number of cells in x, y, and z

        Returns:
            ndarray of [x_centers.size, y_centers.size, z_centers.size]
        """
        return numpy.array([coord.size - 1 for coord in self.exyz], dtype=int)

    @property
    def num_grids(self) -> int:
        """
        The number of grids (number of shifts)
        """
        return self.shifts.shape[0]

    @property
    def cell_data_shape(self):
        """
        The shape of the cell_data ndarray (num_grids, *self.shape).
        """
        return numpy.hstack((self.num_grids, self.shape))

    @property
    def dxyz_with_ghost(self) -> List[numpy.ndarray]:
        """
        Gives dxyz with an additional 'ghost' cell at the end, whose value depends
         on whether or not the axis has periodic boundary conditions. See main description
         above to learn why this is necessary.

         If periodic, final edge shifts same amount as first
         Otherwise, final edge shifts same amount as second-to-last

        Returns:
            list of [dxs, dys, dzs] with each element same length as elements of `self.xyz`
        """
        el = [0 if p else -1 for p in self.periodic]
        return [numpy.hstack((self.dxyz[a], self.dxyz[a][e])) for a, e in zip(range(3), el)]

    @property
    def center(self) -> numpy.ndarray:
        """
        Center position of the entire grid, no shifts applied

        Returns:
            ndarray of [x_center, y_center, z_center]
        """
        # center is just average of first and last xyz, which is just the average of the
        #  first two and last two exyz
        centers = [(self.exyz[a][:2] + self.exyz[a][-2:]).sum() / 4.0 for a in range(3)]
        return numpy.array(centers, dtype=float)

    @property
    def dxyz_limits(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Returns the minimum and maximum cell size for each axis, as a tuple of two 3-element
         ndarrays. No shifts are applied, so these are extreme bounds on these values (as a
         weighted average is performed when shifting).

        Returns:
            Tuple of 2 ndarrays, `d_min=[min(dx), min(dy), min(dz)]` and `d_max=[...]`
        """
        d_min = numpy.array([min(self.dxyz[a]) for a in range(3)], dtype=float)
        d_max = numpy.array([max(self.dxyz[a]) for a in range(3)], dtype=float)
        return d_min, d_max

    def shifted_exyz(self, which_shifts: Optional[int]) -> List[numpy.ndarray]:
        """
        Returns edges for which_shifts.

        Args:
            which_shifts: Which grid (which shifts) to use, or `None` for unshifted

        Returns:
            List of 3 ndarrays of cell edges
        """
        if which_shifts is None:
            return self.exyz
        dxyz = self.dxyz_with_ghost
        shifts = self.shifts[which_shifts, :]

        # If shift is negative, use left cell's dx to determine shift
        for a in range(3):
            if shifts[a] < 0:
                dxyz[a] = numpy.roll(dxyz[a], 1)

        return [self.exyz[a] + dxyz[a] * shifts[a] for a in range(3)]

    def shifted_dxyz(self, which_shifts: Optional[int]) -> List[numpy.ndarray]:
        """
        Returns cell sizes for `which_shifts`.

        Args:
            which_shifts: Which grid (which shifts) to use, or `None` for unshifted

        Returns:
            List of 3 ndarrays of cell sizes
        """
        if which_shifts is None:
            return self.dxyz
        shifts = self.shifts[which_shifts, :]
        dxyz = self.dxyz_with_ghost

        # If shift is negative, use left cell's dx to determine size
        sdxyz = []
        for a in range(3):
            if shifts[a] < 0:
                roll_dxyz = numpy.roll(dxyz[a], 1)
                abs_shift = numpy.abs(shifts[a])
                sdxyz.append(roll_dxyz[:-1] * abs_shift + roll_dxyz[1:] * (1 - abs_shift))
            else:
                sdxyz.append(dxyz[a][:-1] * (1 - shifts[a]) + dxyz[a][1:] * shifts[a])

        return sdxyz

    def shifted_xyz(self, which_shifts: Optional[int]) -> List[numpy.ndarray]:
        """
        Returns cell centers for `which_shifts`.

        Args:
            which_shifts: Which grid (which shifts) to use, or `None` for unshifted

        Returns:
            List of 3 ndarrays of cell centers
        """
        if which_shifts is None:
            return self.xyz
        exyz = self.shifted_exyz(which_shifts)
        dxyz = self.shifted_dxyz(which_shifts)
        return [exyz[a][:-1] + dxyz[a] / 2.0 for a in range(3)]

    def autoshifted_dxyz(self) -> List[numpy.ndarray]:
        """
        Return cell widths, with each dimension shifted by the corresponding shifts.

        Returns:
            `[grid.shifted_dxyz(which_shifts=a)[a] for a in range(3)]`
        """
        if self.num_grids != 3:
            raise GridError('Autoshifting requires exactly 3 grids')
        return [self.shifted_dxyz(which_shifts=a)[a] for a in range(3)]

    def allocate(self, fill_value: Optional[float] = 1.0, dtype=numpy.float32) -> numpy.ndarray:
        """
        Allocate an ndarray for storing grid data.

        Args:
            fill_value: Value to initialize the grid to. If None, an
                uninitialized array is returned.
            dtype: Numpy dtype for the array. Default is `numpy.float32`.

        Returns:
            The allocated array
        """
        if fill_value is None:
            return numpy.empty(self.cell_data_shape, dtype=dtype)
        else:
            return numpy.full(self.cell_data_shape, fill_value, dtype=dtype)

    def __init__(self,
                 pixel_edge_coordinates: Sequence[numpy.ndarray],
                 shifts: numpy.ndarray = Yee_Shifts_E,
                 periodic: Union[bool, Sequence[bool]] = False,
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
        self.exyz = [numpy.unique(pixel_edge_coordinates[i]) for i in range(3)]
        self.shifts = numpy.array(shifts, dtype=float)

        for i in range(3):
            if len(self.exyz[i]) != len(pixel_edge_coordinates[i]):
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

    def save(self: T, filename: str) -> T:
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

    def copy(self: T) -> T:
        """
        Returns:
            Deep copy of the grid.
        """
        return copy.deepcopy(self)
