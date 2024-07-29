from typing import Protocol

import numpy
from numpy.typing import NDArray

from . import GridError


class GridBase(Protocol):
    exyz: list[NDArray]
    """Cell edges. Monotonically increasing without duplicates."""

    periodic: list[bool]
    """For each axis, determines how far the rightmost boundary gets shifted. """

    shifts: NDArray
    """Offsets `[[x0, y0, z0], [x1, y1, z1], ...]` for grid `0,1,...`"""

    @property
    def dxyz(self) -> list[NDArray]:
        """
        Cell sizes for each axis, no shifts applied

        Returns:
            List of 3 ndarrays of cell sizes
        """
        return [numpy.diff(ee) for ee in self.exyz]

    @property
    def xyz(self) -> list[NDArray]:
        """
        Cell centers for each axis, no shifts applied

        Returns:
            List of 3 ndarrays of cell edges
        """
        return [self.exyz[a][:-1] + self.dxyz[a] / 2.0 for a in range(3)]

    @property
    def shape(self) -> NDArray[numpy.intp]:
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
    def cell_data_shape(self) -> NDArray[numpy.intp]:
        """
        The shape of the cell_data ndarray (num_grids, *self.shape).
        """
        return numpy.hstack((self.num_grids, self.shape))

    @property
    def dxyz_with_ghost(self) -> list[NDArray]:
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
        return [numpy.hstack((self.dxyz[a], self.dxyz[a][e])) for a, e in zip(range(3), el, strict=True)]

    @property
    def center(self) -> NDArray[numpy.float64]:
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
    def dxyz_limits(self) -> tuple[NDArray, NDArray]:
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

    def shifted_exyz(self, which_shifts: int | None) -> list[NDArray]:
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

    def shifted_dxyz(self, which_shifts: int | None) -> list[NDArray]:
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

    def shifted_xyz(self, which_shifts: int | None) -> list[NDArray[numpy.float64]]:
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

    def autoshifted_dxyz(self) -> list[NDArray[numpy.float64]]:
        """
        Return cell widths, with each dimension shifted by the corresponding shifts.

        Returns:
            `[grid.shifted_dxyz(which_shifts=a)[a] for a in range(3)]`
        """
        if self.num_grids != 3:
            raise GridError('Autoshifting requires exactly 3 grids')
        return [self.shifted_dxyz(which_shifts=a)[a] for a in range(3)]

    def allocate(self, fill_value: float | None = 1.0, dtype: type[numpy.number] = numpy.float32) -> NDArray:
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
        return numpy.full(self.cell_data_shape, fill_value, dtype=dtype)
