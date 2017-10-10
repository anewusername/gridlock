"""
Position-related methods for Grid class
"""

from typing import List

import numpy
from numpy import zeros

from . import GridError


def ind2pos(self,
            ind: numpy.ndarray or List,
            which_shifts: int = None,
            round_ind: bool = True,
            check_bounds: bool = True
            ) -> numpy.ndarray:
    """
    Returns the natural position corresponding to the specified cell center indices.
     The resulting position is clipped to the bounds of the grid
    (to cell centers if round_ind=True, or cell outer edges if round_ind=False)

    :param ind: Indices of the position. Can be fractional. (3-element ndarray or list)
    :param which_shifts: which grid number (shifts) to use
    :param round_ind: Whether to round ind to the nearest integer position before indexing
            (default True)
    :param check_bounds: Whether to raise an GridError if the provided ind is outside of
            the grid, as defined above (centers if round_ind, else edges) (default True)
    :return: 3-element ndarray specifying the natural position
    :raises: GridError
    """
    if which_shifts is not None and which_shifts >= self.shifts.shape[0]:
        raise GridError('Invalid shifts')
    ind = numpy.array(ind, dtype=float)

    if check_bounds:
        if round_ind:
            low_bound = 0.0
            high_bound = -1
        else:
            low_bound = -0.5
            high_bound = -0.5
        if (ind < low_bound).any() or (ind > self.shape - high_bound).any():
            raise GridError('Position outside of grid: {}'.format(ind))

    if round_ind:
        rind = numpy.clip(numpy.round(ind).astype(int), 0, self.shape - 1)
        sxyz = self.shifted_xyz(which_shifts)
        position = [sxyz[a][rind[a]].astype(int) for a in range(3)]
    else:
        sexyz = self.shifted_exyz(which_shifts)
        position = [numpy.interp(ind[a], numpy.arange(sexyz[a].size) - 0.5, sexyz[a])
                    for a in range(3)]
    return numpy.array(position, dtype=float)


def pos2ind(self,
            r: numpy.ndarray or List,
            which_shifts: int or None,
            round_ind: bool=True,
            check_bounds: bool=True
            ) -> numpy.ndarray:
    """
    Returns the cell-center indices corresponding to the specified natural position.
         The resulting position is clipped to within the outer centers of the grid.

    :param r: Natural position that we will convert into indices (3-element ndarray or list)
    :param which_shifts: which grid number (shifts) to use
    :param round_ind: Whether to round the returned indices to the nearest integers.
    :param check_bounds: Whether to throw an GridError if r is outside the grid edges
    :return: 3-element ndarray specifying the indices
    :raises: GridError
    """
    r = numpy.squeeze(r)
    if r.size != 3:
        raise GridError('r must be 3-element vector: {}'.format(r))

    if (which_shifts is not None) and (which_shifts >= self.shifts.shape[0]):
        raise GridError('Invalid which_shifts: {}'.format(which_shifts))

    sexyz = self.shifted_exyz(which_shifts)

    if check_bounds:
        for a in range(3):
            if self.shape[a] > 1 and (r[a] < sexyz[a][0] or r[a] > sexyz[a][-1]):
                raise GridError('Position[{}] outside of grid!'.format(a))

    grid_pos = zeros((3,))
    for a in range(3):
        xi = numpy.digitize(r[a], sexyz[a]) - 1 # Figure out which cell we're in
        xi_clipped = numpy.clip(xi, 0, sexyz[a].size - 2)  # Clip back into grid bounds

        # No need to interpolate if round_ind is true or we were outside the grid
        if round_ind or xi != xi_clipped:
            grid_pos[a] = xi_clipped
        else:
            # Interpolate
            x = self.shifted_xyz(which_shifts)[a][xi]
            dx = self.shifted_dxyz(which_shifts)[a][xi]
            f = (r[a] - x) / dx
            
            # Clip to centers
            grid_pos[a] = numpy.clip(xi + f, 0, self.shape[a] - 1)
    return grid_pos
