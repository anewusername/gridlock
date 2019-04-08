from typing import List, Tuple, Callable, Dict

import numpy
from numpy import diff, floor, ceil, zeros, hstack, newaxis

import pickle
import warnings
import copy

from . import GridError, Direction
from ._helpers import is_scalar


__author__ = 'Jan Petykiewicz'

eps_callable_type = Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray], numpy.ndarray]


class Grid(object):
    """
    Simulation grid generator intended for electromagnetic simulations.
    Can be used to generate non-uniform rectangular grids (the entire grid
    is generated based on the coordinates of the boundary points). Also does
    straightforward natural <-> grid unit conversion.

    self.grids[i][a,b,c] contains the value of epsilon for the cell located around
          (xyz[0][a] + dxyz[0][a] * shifts[i, 0],
           xyz[1][b] + dxyz[1][b] * shifts[i, 1],
           xyz[2][c] + dxyz[2][c] * shifts[i, 2]).
     You can get raw edge coordinates (exyz),
                   center coordinates (xyz),
                           cell sizes (dxyz),
      from the properties named as above, or get them for a given grid by using the
      self.shifted_*xyz(which_shifts) functions.

     The sizes of adjacent cells are taken into account when applying shifts. The
      total shift for each edge is chosen using (shift * dx_of_cell_being_moved_through).

     It is tricky to determine the size of the right-most cell after shifting,
      since its right boundary should shift by shifts[i][a] * dxyz[a][dxyz[a].size],
      where the dxyz element refers to a cell that does not exist.
     Because of this, we either assume this 'ghost' cell is the same size as the last
      real cell, or, if self.periodic[a] is set to True, the same size as the first cell.
    """

    # Cell edges. Monotonically increasing without duplicates
    exyz = []               # type: List[numpy.ndarray]

    # epsilon (or mu, or whatever) grids
    grids = []              # type: List[numpy.ndarray]

    # [[x0 y0 z0], [x1, y1, z1], ...] offsets for grid 0,1,...
    shifts = None           # type: numpy.ndarray

    # For each axis, determines how far the rightmost boundary gets shifted
    periodic = [False] * 3  # type: List[bool]

    # Intended for use as static constants
    Yee_Shifts_E = 0.5 * numpy.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]], dtype=float)    # type: numpy.ndarray
    Yee_Shifts_H = 0.5 * numpy.array([[0, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 0]], dtype=float)    # type: numpy.ndarray

    from .draw import draw_polygons, draw_polygon, draw_slab, draw_cuboid, \
           draw_cylinder, draw_extrude_rectangle
    from .read import get_slice, visualize_slice, visualize_isosurface
    from .position import ind2pos, pos2ind

    @property
    def dxyz(self) -> List[numpy.ndarray]:
        """
        Cell sizes for each axis, no shifts applied

        :return: List of 3 ndarrays of cell sizes
        """
        return [diff(self.exyz[a]) for a in range(3)]

    @property
    def xyz(self) -> List[numpy.ndarray]:
        """
        Cell centers for each axis, no shifts applied

        :return: List of 3 ndarrays of cell edges
        """
        return [self.exyz[a][:-1] + self.dxyz[a] / 2.0 for a in range(3)]

    @property
    def shape(self) -> numpy.ndarray:
        """
        The number of cells in x, y, and z

        :return: ndarray [x_centers.size, y_centers.size, z_centers.size]
        """
        return numpy.array([coord.size - 1 for coord in self.exyz], dtype=int)

    @property
    def dxyz_with_ghost(self) -> List[numpy.ndarray]:
        """
        Gives dxyz with an additional 'ghost' cell at the end, whose value depends
         on whether or not the axis has periodic boundary conditions. See main description
         above to learn why this is necessary.

         If periodic, final edge shifts same amount as first
         Otherwise, final edge shifts same amount as second-to-last

        :return: list of [dxs, dys, dzs] with each element same length as elements of self.xyz
        """
        el = [0 if p else -1 for p in self.periodic]
        return [hstack((self.dxyz[a], self.dxyz[a][e])) for a, e in zip(range(3), el)]

    @property
    def center(self) -> numpy.ndarray:
        """
        Center position of the entire grid, no shifts applied
        :return: ndarray [x_center, y_center, z_center]
        """
        # center is just average of first and last xyz, which is just the average of the
        #  first two and last two exyz
        centers = [(self.exyz[a][:2] + self.exyz[a][-2:]) / 4.0 for a in range(3)]
        return numpy.array(centers, dtype=float)

    @property
    def dxyz_limits(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Returns the minimum and maximum cell size for each axis, as a tuple of two 3-element
         ndarrays. No shifts are applied, so these are extreme bounds on these values (as a
         weighted average is performed when shifting).

        :return: List of 2 ndarrays, d_min=[min(dx), min(dy), min(dz)] and d_max=[...]
        """
        d_min = numpy.array([min(self.dxyz[a]) for a in range(3)], dtype=float)
        d_max = numpy.array([max(self.dxyz[a]) for a in range(3)], dtype=float)
        return d_min, d_max

    def shifted_exyz(self, which_shifts: int or None) -> List[numpy.ndarray]:
        """
        Returns edges for which_shifts.

        :param which_shifts: Which grid (which shifts) to use, or None for unshifted
        :return: List of 3 ndarrays of cell edges
        """
        if which_shifts is None:
            return self.exyz
        dxyz = self.dxyz_with_ghost
        shifts = self.shifts[which_shifts, :]

        # If shift is negative, use left cell's dx to determine shift
        for a in range(3):
            if shifts[a] < 0:
                dxyz[a] = numpy.roll(dxyz, 1)

        return [self.exyz[a] + dxyz[a] * shifts[a] for a in range(3)]

    def shifted_dxyz(self, which_shifts: int or None) -> List[numpy.ndarray]:
        """
        Returns cell sizes for which_shifts.

        :param which_shifts: Which grid (which shifts) to use, or None for unshifted
        :return: List of 3 ndarrays of cell sizes
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

    def shifted_xyz(self, which_shifts: int or None) -> List[numpy.ndarray]:
        """
        Returns cell centers for which_shifts.

        :param which_shifts: Which grid (which shifts) to use, or None for unshifted
        :return: List of 3 ndarrays of cell centers
        """
        if which_shifts is None:
            return self.xyz
        exyz = self.shifted_exyz(which_shifts)
        dxyz = self.shifted_dxyz(which_shifts)
        return [exyz[a][:-1] + dxyz[a] / 2.0 for a in range(3)]

    def autoshifted_dxyz(self):
        """
        Return cell widths, with each dimension shifted by the corresponding shifts.

        :return: [grid.shifted_dxyz(which_shifts=a)[a] for a in range(3)]
        """
        if len(self.grids) != 3:
            raise GridError('autoshifting requires exactly 3 grids')
        return [self.shifted_dxyz(which_shifts=a)[a] for a in range(3)]


    def __init__(self,
                 pixel_edge_coordinates: List[List or numpy.ndarray],
                 shifts: numpy.ndarray or List = Yee_Shifts_E,
                 initial: float or numpy.ndarray or List[float] or List[numpy.ndarray] = (1.0,)*3,
                 num_grids: int = None,
                 periodic: bool or List[bool] = False):
        """
        Initialize a new Grid

        :param pixel_edge_coordinates: 3-element list of (ndarrays or lists) specifying the
         coordinates of the pixel edges in each dimensions
         (ie, [[x0, x1, x2,...], [y0,...], [z0,...]] where the first pixel has x-edges x=x0 and
          x=x1, the second has edges x=x1 and x=x2, etc.)
        :param shifts: Nx3 array containing [x, y, z] offsets for each of N grids.
         E-field Yee shifts are used by default.
        :param initial: Grids are initialized to this value. If scalar, all grids are initialized
         with ndarrays full of the scalar. If a list of scalars, grid[i] is initialized to an
         ndarray full of initial[i]. If a list of ndarrays of the same shape as the grids, grid[i]
         is set to initial[i]. Default 1.
        :param num_grids: How many grids to create. Must be <= shifts.shape[0].
         Default is shifts.shape[0]
        :param periodic: Specifies how the sizes of edge cells are calculated; see main class
         documentation. List of 3 bool, or a single bool that gets broadcast. Default False.
        :raises: GridError
        """
        self.exyz = [numpy.unique(pixel_edge_coordinates[i]) for i in range(3)]
        for i in range(3):
            if len(self.exyz[i]) != len(pixel_edge_coordinates[i]):
                warnings.warn('Dimension {} had duplicate edge coordinates'.format(i), stacklevel=2)

        if is_scalar(periodic):
            periodic = [periodic] * 3
        self.periodic = periodic

        self.shifts = numpy.array(shifts, dtype=float)
        if len(self.shifts.shape) != 2:
            raise GridError('Misshapen shifts: shifts must have two axes! '
                            ' The given shifts has shape {}'.format(self.shifts.shape))
        if self.shifts.shape[1] != 3:
            raise GridError('Misshapen shifts; second axis size should be 3,'
                            ' shape is {}'.format(self.shifts.shape))

        if (numpy.abs(self.shifts) > 1).any():
            raise GridError('Only shifts in the range [-1, 1] are currently supported')

        if (self.shifts < 0).any():
            # TODO: Test negative shifts
            warnings.warn('Negative shifts are still experimental and mostly untested, be careful!', stacklevel=2)

        num_shifts = self.shifts.shape[0]
        if num_grids is None:
            num_grids = num_shifts
        elif num_grids > num_shifts:
            raise GridError('Number of grids exceeds number of shifts (%u)' % num_shifts)

        grids_shape = hstack((num_grids, self.shape))
        if is_scalar(initial):
            if isinstance(initial, int):
                warnings.warn('Initial value is an int, grids will be integer-typed!', stacklevel=2)
            self.grids = numpy.full(grids_shape, initial)
        else:
            if len(initial) < num_grids:
                raise GridError('Too few initial grids specified!')

            self.grids = [None] * num_grids
            for i in range(num_grids):
                if is_scalar(initial[i]):
                    if initial[i] is not None:
                        if isinstance(initial[i], int):
                            warnings.warn('Initial value is an int, grid {} will be integer-typed!'.format(i), stacklevel=2)
                        self.grids[i] = numpy.full(self.shape, initial[i])
                else:
                    if not numpy.array_equal(initial[i].shape, self.shape):
                        raise GridError('Initial grid sizes must match given coordinates')
                    self.grids[i] = initial[i]

    @staticmethod
    def load(filename: str) -> 'Grid':
        """
        Load a grid from a file

        :param filename: Filename to load from.
        """
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)

        g = Grid([[-1, 1]] * 3)
        g.__dict__.update(tmp_dict)
        return g

    def save(self, filename: str):
        """
        Save to file.

        :param filename: Filename to save to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=2)

    def copy(self):
        """
        Return a deep copy of the grid.

        :return: Deep copy of the grid.
        """
        return copy.deepcopy(self)

