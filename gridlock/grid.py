from typing import List, Tuple, Callable, Dict

import numpy
from numpy import diff, floor, ceil, zeros, hstack, newaxis

import pickle
import warnings
import copy

from float_raster import raster

# .visualize_* uses matplotlib
# .visualize_isosurface uses skimage
# .visualize_isosurface uses mpl_toolkits.mplot3d

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

    self.grids[i][a,b,c] contains the value of epsilon for the cell located at
          (xyz[0][a]+dxyz[0][a]*shifts[i, 0],
           xyz[1][b]+dxyz[1][b]*shifts[i, 1],
           xyz[2][c]+dxyz[2][c]*shifts[i, 2]).
     You can get raw edge coordinates (exyz),
                   center coordinates (xyz),
                           cell sizes (dxyz),
      from the properties named as above, or get them for a given grid by using the
      self.shifted_*xyz(which_shifts) functions.

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
        centers = [(self.exyz[a][:1] + self.exyz[a][-1:]) / 4.0 for a in range(3)]
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
        return [(dxyz[a][:-1] * (1 - shifts[a]) + dxyz[a][1:] * shifts[a]) for a in range(3)]

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

    def ind2pos(self,
                ind: numpy.ndarray or List,
                which_shifts: int = None,
                round_ind: bool = True,
                check_bounds: bool = True
                ) -> numpy.ndarray:
        """
        Returns the natural position corresponding to the specified indices.
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
            rind = numpy.clip(numpy.round(ind), 0, self.shape - 1)
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
        Returns the indices corresponding to the specified natural position.
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
            raise GridError('')

        sexyz = self.shifted_exyz(which_shifts)

        if check_bounds:
            for a in range(3):
                if self.shape[a] > 1 and (r[a] < sexyz[a][0] or r[a] > sexyz[a][-1]):
                    raise GridError('Position[{}] outside of grid!'.format(a))

        grid_pos = zeros((3,))
        for a in range(3):
            xi = numpy.digitize(r[a], sexyz[a])  # Figure out which cell we're in
            xi_clipped = numpy.clip(xi, 1, sexyz[a].size - 1) - 1  # Clip back into grid bounds

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
                warnings.warn('Dimension {} had duplicate edge coordinates'.format(i))

        if is_scalar(periodic):
            periodic = [periodic] * 3
        self.periodic = periodic

        self.shifts = numpy.array(shifts, dtype=float)
        if self.shifts.shape[1] != 3:
            GridError('Misshapen shifts; second axis size should be 3,'
                      ' shape is {}'.format(self.shifts.shape))

        num_shifts = self.shifts.shape[0]
        if num_grids is None:
            num_grids = num_shifts
        elif num_grids > num_shifts:
            raise GridError('Number of grids exceeds number of shifts (%u)' % num_shifts)

        grids_shape = hstack((num_grids, self.shape))
        if is_scalar(initial):
            self.grids = numpy.full(grids_shape, initial)
        else:
            if len(initial) < num_grids:
                raise GridError('Too few initial grids specified!')

            self.grids = [None] * num_grids
            for i in range(num_grids):
                if is_scalar(initial[i]):
                    if initial[i] is not None:
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

    def draw_polygons(self,
                      surface_normal: Direction or int,
                      center: List or numpy.ndarray,
                      polygons: List[numpy.ndarray or List],
                      thickness: float,
                      eps: List[float or eps_callable_type] or float or eps_callable_type):
        """
        Draw polygons on an axis-aligned plane.

        :param surface_normal: Axis normal to the plane we're drawing on. Can be a Direction or
         integer in range(3)
        :param center: 3-element ndarray or list specifying an offset applied to all the polygons
        :param polygons: List of Nx2 or Nx3 ndarrays, each specifying the vertices of a polygon
             (non-closed, clockwise). If Nx3, the surface_normal coordinate is ignored. Each polygon
             must have at least 3 vertices.
        :param thickness: Thickness of the layer to draw
        :param eps: Value to draw with ('epsilon'). Can be scalar, callable, or a list
             of any of these (1 per grid). Callable values should take ndarrays x, y, z of equal
             shape and return an ndarray of equal shape containing the eps value at the given x, y,
             and z (natural, not grid coordinates).
        :raises: GridError
        """
        # Turn surface_normal into its integer representation
        if isinstance(surface_normal, Direction):
            surface_normal = surface_normal.value

        if surface_normal not in range(3):
            raise GridError('Invalid surface_normal direction')

        center = numpy.squeeze(center)

        # Check polygons, and remove redundant coordinates
        surface = numpy.delete(range(3), surface_normal)

        for i, polygon in enumerate(polygons):
            malformed = 'Malformed polygon: (%i)' % i
            if polygon.shape[1] not in (2, 3):
                    raise GridError(malformed + 'must be a Nx2 or Nx3 ndarray')
            if polygon.shape[1] == 3:
                polygon = polygon[surface, :]

            if not polygon.shape[0] > 2:
                raise GridError(malformed + 'must consist of more than 2 points')
            if polygon.ndim > 2 and not numpy.unique(polygon[:, surface_normal]).size == 1:
                raise GridError(malformed + 'must be in plane with surface normal %s'
                                % 'xyz'[surface_normal])

        # Broadcast eps where necessary
        if is_scalar(eps):
            eps = [eps] * len(self.grids)
        elif isinstance(eps, numpy.ndarray):
            raise GridError('ndarray not supported for eps')

        # ## Compute sub-domain of the grid occupied by polygons
        # 1) Compute outer bounds (bd) of polygons
        bd_2d_min = [0, 0]
        bd_2d_max = [0, 0]
        for polygon in polygons:
            bd_2d_min = numpy.minimum(bd_2d_min, polygon.min(axis=0))
            bd_2d_max = numpy.maximum(bd_2d_max, polygon.max(axis=0))
        bd_min = numpy.insert(bd_2d_min, surface_normal, -thickness / 2.0) + center
        bd_max = numpy.insert(bd_2d_max, surface_normal, +thickness / 2.0) + center

        # 2) Find indices (bdi) just outside bd elements
        buf = 2  # size of safety buffer
        # Use s_min and s_max with unshifted pos2ind to get absolute limits on
        #  the indices the polygons might affect
        s_min = self.shifts.min(axis=0)
        s_max = self.shifts.max(axis=0)
        bdi_min = self.pos2ind(bd_min + s_min, None, round_ind=False, check_bounds=False) - buf
        bdi_max = self.pos2ind(bd_max + s_max, None, round_ind=False, check_bounds=False) + buf
        bdi_min = numpy.maximum(floor(bdi_min), 0).astype(int)
        bdi_max = numpy.minimum(ceil(bdi_max), self.shape - 1).astype(int)

        # 3) Adjust polygons for center
        polygons = [poly + center[surface] for poly in polygons]

        # iterate over grids
        for (i, grid) in enumerate(self.grids):
            # ## Evaluate or expand eps[i]
            if callable(eps[i]):
                # meshgrid over the (shifted) domain
                domain = [self.shifted_xyz(i)[k][bdi_min[k]:bdi_max[k]+1] for k in range(3)]
                (x0, y0, z0) = numpy.meshgrid(*domain, indexing='ij')

                # evaluate on the meshgrid
                eps[i] = eps[i](x0, y0, z0)
                if not numpy.isfinite(eps[i]).all():
                    raise GridError('Non-finite values in eps[%u]' % i)
            elif not is_scalar(eps[i]):
                raise GridError('Unsupported eps[{}]: {}'.format(i, type(eps[i])))
            # do nothing if eps[i] is scalar non-callable

            # ## Generate weighing function
            def to_3d(vector: List or numpy.ndarray, val: float=0.0):
                return numpy.insert(vector, surface_normal, (val,))

            w_xy = zeros((bdi_max - bdi_min + 1)[surface].astype(int))

            # Draw each polygon separately
            for polygon in polygons:

                # Get the boundaries of the polygon
                pbd_min = polygon.min(axis=0)
                pbd_max = polygon.max(axis=0)

                # Find indices in w_xy just outside polygon
                #  using per-grid xy-weights (self.shifted_xyz())
                corner_min = self.pos2ind(to_3d(pbd_min), i,
                                          check_bounds=False)[surface].astype(int)
                corner_max = self.pos2ind(to_3d(pbd_max), i,
                                          check_bounds=False)[surface].astype(int)

                # Find indices in w_xy which are modified by polygon
                # First for the edge coordinates (+1 since we're indexing edges)
                edge_slices = [numpy.s_[i:f + 2] for i, f in zip(corner_min, corner_max)]
                # Then for the pixel centers (-bdi_min since we're
                #  calculating weights within a subspace)
                centers_slice = [numpy.s_[i:f + 1] for i, f in zip(corner_min - bdi_min[surface],
                                                                   corner_max - bdi_min[surface])]

                aa_x, aa_y = (self.shifted_exyz(i)[a][s] for a, s in zip(surface, edge_slices))
                w_xy[centers_slice] += raster(polygon.T, aa_x, aa_y)

            # Clamp overlapping polygons to 1
            w_xy = numpy.minimum(w_xy, 1.0)

            # 2) Generate weights in z-direction
            w_z = numpy.zeros(((bdi_max - bdi_min + 1)[surface_normal], ))

            def get_zi(offset):
                pos_3d = to_3d([0, 0], center[surface_normal] + offset)
                grid_coords = self.pos2ind(pos_3d, i, check_bounds=False, round_ind=False)
                w_coord_fp = (grid_coords - bdi_min)[surface_normal]
                w_coord = floor(w_coord_fp).astype(int)
                return w_coord_fp, w_coord

            zi_top_fp, zi_top = get_zi(+thickness/2.0)
            zi_bot_fp, zi_bot = get_zi(-thickness/2.0)

            w_z[zi_bot:zi_top + 1] = 1

            if zi_top_fp != zi_top < self.shape[surface_normal] - 1:
                f = zi_top_fp - zi_top
                w_z[zi_top] = f
            if zi_bot_fp != zi_bot > 0:
                f = zi_bot_fp - zi_bot
                w_z[zi_bot] = 1 - f

            # 3) Generate total weight function
            w = (w_xy[:, :, newaxis] * w_z).transpose(numpy.insert([0, 1], surface_normal, (2,)))

            # ## Modify the grid
            g_slice = [numpy.s_[bdi_min[a]:bdi_max[a] + 1] for a in range(3)]
            self.grids[i][g_slice] = (1 - w) * self.grids[i][g_slice] + w * eps[i]

    def draw_polygon(self,
                     surface_normal: Direction or int,
                     center: List or numpy.ndarray,
                     polygon: List or numpy.ndarray,
                     thickness: float,
                     eps: List[float or eps_callable_type] or float or eps_callable_type):
        """
        Draw a polygon on an axis-aligned plane.

        :param surface_normal: Axis normal to the plane we're drawing on. Can be a Direction or
         integer in range(3)
        :param center: 3-element ndarray or list specifying an offset applied to the polygon
        :param polygon: Nx2 or Nx3 ndarray specifying the vertices of a polygon (non-closed,
             clockwise). If Nx3, the surface_normal coordinate is ignored. Must have at least 3
             vertices.
        :param thickness: Thickness of the layer to draw
        :param eps: Value to draw with ('epsilon'). See draw_polygons() for details.
        """
        self.draw_polygons(surface_normal, center, [polygon], thickness, eps)

    def draw_slab(self,
                  surface_normal: Direction or int,
                  center: List or numpy.ndarray,
                  thickness: float,
                  eps: List[float or eps_callable_type] or float or eps_callable_type):
        """
        Draw an axis-aligned infinite slab.

        :param surface_normal: Axis normal to the plane we're drawing on. Can be a Direction or
         integer in range(3)
        :param center: Surface_normal coordinate at the center of the slab
        :param thickness: Thickness of the layer to draw
        :param eps: Value to draw with ('epsilon'). See draw_polygons() for details.
        """
        # Turn surface_normal into its integer representation
        if isinstance(surface_normal, Direction):
            surface_normal = surface_normal.value
        if surface_normal not in range(3):
            raise GridError('Invalid surface_normal direction')

        if not is_scalar(center):
            center = numpy.squeeze(center)
            if len(center) == 3:
                center = center[surface_normal]
            else:
                raise GridError('Bad center: {}'.format(center))

        # Find center of slab
        center_shift = self.center
        center_shift[surface_normal] = center

        surface = numpy.delete(range(3), surface_normal)

        xyz_min = numpy.array([self.xyz[a][0] for a in range(3)], dtype=float)[surface]
        xyz_max = numpy.array([self.xyz[a][-1] for a in range(3)], dtype=float)[surface]

        dxyz = numpy.array([max(self.dxyz[i]) for i in surface], dtype=float)

        xyz_min -= 4 * dxyz
        xyz_max += 4 * dxyz

        p = numpy.array([[xyz_min[0], xyz_max[1]],
                         [xyz_max[0], xyz_max[1]],
                         [xyz_max[0], xyz_min[1]],
                         [xyz_min[0], xyz_min[1]]], dtype=float)

        self.draw_polygon(surface_normal, center_shift, p, thickness, eps)

    # TODO: TEST ME!
    def draw_cuboid(self,
                    center: List or numpy.ndarray,
                    dimensions: List or numpy.ndarray,
                    eps: List[float or eps_callable_type] or float or eps_callable_type):
        """
        Draw an axis-aligned cuboid

        :param center: 3-element ndarray or list specifying the cylinder's center
        :param dimensions: 3-element list or ndarray containing the x, y, and z edge-to-edge
            sizes of the cuboid
        :param eps: Value to draw with ('epsilon'). See draw_polygons() for details.
        """
        p = numpy.array([[-dimensions[0], +dimensions[1]],
                         [+dimensions[0], +dimensions[1]],
                         [+dimensions[0], -dimensions[1]],
                         [-dimensions[0], -dimensions[1]]], dtype=float) / 2
        thickness = dimensions[2]
        self.draw_polygon(Direction.z, center, p, thickness, eps)

    def draw_cylinder(self,
                      surface_normal: Direction or int,
                      center: List or numpy.ndarray,
                      radius: float,
                      thickness: float,
                      num_points: int,
                      eps: List[float or eps_callable_type] or float or eps_callable_type):
        """
        Draw an axis-aligned cylinder. Approximated by a num_points-gon

        :param surface_normal: Axis normal to the plane we're drawing on. Can be a Direction or
         integer in range(3)
        :param center: 3-element ndarray or list specifying the cylinder's center
        :param radius: cylinder radius
        :param thickness: Thickness of the layer to draw
        :param num_points: The circle is approximated by a polygon with num_points vertices
        :param eps: Value to draw with ('epsilon'). See draw_polygons() for details.
        """
        theta = numpy.linspace(0, 2*numpy.pi, num_points, endpoint=False)
        x = radius * numpy.sin(theta)
        y = radius * numpy.cos(theta)
        polygon = hstack((x[:, newaxis], y[:, newaxis]))
        self.draw_polygon(surface_normal, center, polygon, thickness, eps)

    def draw_extrude_rectangle(self,
                               rectangle: List or numpy.ndarray,
                               direction: Direction or int,
                               polarity: int,
                               distance: float):
        """
        Extrude a rectangle of a previously-drawn structure along an axis.

        :param rectangle: 2x3 ndarray or list specifying the rectangle's corners
        :param direction: Direction to extrude in. Direction enum or int in range(3)
        :param polarity: +1 or -1, direction along axis to extrude in
        :param distance: How far to extrude
        """
        # Turn extrude_direction into its integer representation
        if isinstance(direction, Direction):
            direction = direction.value
        if abs(direction) not in range(3):
            raise GridError('Invalid extrude_direction')

        s = numpy.sign(polarity)
        surface = numpy.delete(range(3), direction)

        rectangle = numpy.array(rectangle, dtype=float)
        if s == 0:
            raise GridError('0 is not a valid polarity')
        if direction not in range(3):
            raise GridError('Invalid direction: {}'.format(direction))
        if rectangle[0, direction] != rectangle[1, direction]:
            raise GridError('Rectangle entries along extrusion direction do not match.')

        center = rectangle.sum(axis=0) / 2.0
        center[direction] += s * distance / 2.0

        dim = numpy.fabs(diff(rectangle, axis=0).T)[surface]
        p = numpy.vstack((numpy.array([-1, -1, 1, 1], dtype=float) * dim[0]/2.0,
                          numpy.array([-1, 1, 1, -1], dtype=float) * dim[1]/2.0)).T
        thickness = distance

        eps_func = [None] * len(self.grids)
        for i, grid in enumerate(self.grids):
            z = self.pos2ind(rectangle[0, :], i, round_ind=False, check_bounds=False)[direction]

            ind = [int(floor(z)) if i == direction else slice(None) for i in range(3)]

            fpart = z - floor(z)
            mult = [1-fpart, fpart][::s]  # reverses if s negative

            eps = mult[0] * grid[ind]
            ind[direction] += 1
            eps += mult[1] * grid[ind]

            def f_eps(xs, ys, zs):
                # transform from natural position to index
                xyzi = numpy.array([self.pos2ind(qrs, which_shifts=i)
                                    for qrs in zip(xs.flat, ys.flat, zs.flat)], dtype=int)
                # reshape to original shape and keep only in-plane components
                (qi, ri) = [numpy.reshape(xyzi[:, k], xs.shape) for k in surface]
                return eps[qi, ri]

            eps_func[i] = f_eps

        self.draw_polygon(direction, center, p, thickness, eps_func)

    def get_slice(self,
                  surface_normal: Direction or int,
                  center: float,
                  which_shifts: int = 0,
                  sample_period: int = 1
                  ) -> numpy.ndarray:
        """
            Retrieve a slice of a grid.
            Interpolates if given a position between two planes.

            :param surface_normal: Axis normal to the plane we're displaying. Can be a Direction or
             integer in range(3)
            :param center: Scalar specifying position along surface_normal axis.
            :param which_shifts: Which grid to display. Default is the first grid (0).
            :param sample_period: Period for down-sampling the image. Default 1 (disabled)
            :return Array containing the portion of the grid.
        """
        if not is_scalar(center) and numpy.isreal(center):
            raise GridError('center must be a real scalar')

        sp = round(sample_period)
        if sp <= 0:
            raise GridError('sample_period must be positive')

        if not is_scalar(which_shifts) or which_shifts < 0:
            raise GridError('Invalid which_shifts')

        # Turn surface_normal into its integer representation
        if isinstance(surface_normal, Direction):
            surface_normal = surface_normal.value
        if surface_normal not in range(3):
            raise GridError('Invalid surface_normal direction')

        surface = numpy.delete(range(3), surface_normal)

        # Extract indices and weights of planes
        center3 = numpy.insert([0, 0], surface_normal, (center,))
        center_index = self.pos2ind(center3, which_shifts,
                                    round_ind=False, check_bounds=False)[surface_normal]
        centers = numpy.unique([floor(center_index), ceil(center_index)]).astype(int)
        if len(centers) == 2:
            fpart = center_index - floor(center_index)
            w = [1 - fpart, fpart]  # longer distance -> less weight
        else:
            w = [1]

        c_min, c_max = (self.xyz[surface_normal][i] for i in [0, -1])
        if center < c_min or center > c_max:
            raise GridError('Coordinate of selected plane must be within simulation domain')

        # Extract grid values from planes above and below visualized slice
        sliced_grid = zeros(self.shape[surface])
        for ci, weight in zip(centers, w):
            s = tuple(ci if a == surface_normal else numpy.s_[::sp] for a in range(3))
            sliced_grid += weight * self.grids[which_shifts][tuple(s)]

        # Remove extra dimensions
        sliced_grid = numpy.squeeze(sliced_grid)

        return sliced_grid

    def visualize_slice(self,
                        surface_normal: Direction or int,
                        center: float,
                        which_shifts: int = 0,
                        sample_period: int = 1,
                        finalize: bool = True,
                        pcolormesh_args: Dict = None):
        """
        Visualize a slice of a grid.
        Interpolates if given a position between two planes.

        :param surface_normal: Axis normal to the plane we're displaying. Can be a Direction or
         integer in range(3)
        :param center: Scalar specifying position along surface_normal axis.
        :param which_shifts: Which grid to display. Default is the first grid (0).
        :param sample_period: Period for down-sampling the image. Default 1 (disabled)
        :param finalize: Whether to call pyplot.show() after constructing the plot. Default True
        """
        from matplotlib import pyplot

        # Set surface normal to its integer value
        if isinstance(surface_normal, Direction):
            surface_normal = surface_normal.value

        if pcolormesh_args is None:
            pcolormesh_args = {}

        grid_slice = self.get_slice(surface_normal=surface_normal,
                                    center=center,
                                    which_shifts=which_shifts,
                                    sample_period=sample_period)

        surface = numpy.delete(range(3), surface_normal)

        x, y = (self.shifted_exyz(which_shifts)[a] for a in surface)
        xmesh, ymesh = numpy.meshgrid(x, y, indexing='ij')
        x_label, y_label = ('xyz'[a] for a in surface)

        pyplot.figure()
        pyplot.pcolormesh(xmesh, ymesh, grid_slice, **pcolormesh_args)
        pyplot.colorbar()
        pyplot.gca().set_aspect('equal', adjustable='box')
        pyplot.xlabel(x_label)
        pyplot.ylabel(y_label)
        if finalize:
            pyplot.show()

    def visualize_isosurface(self,
                             level: float = None,
                             which_shifts: int = 0,
                             sample_period: int = 1,
                             show_edges: bool = True,
                             finalize: bool = True):
        """
        Draw an isosurface plot of the device.

        :param level: Value at which to find isosurface. Default (None) uses mean value in grid.
        :param which_shifts: Which grid to display. Default is the first grid (0).
        :param sample_period: Period for down-sampling the image. Default 1 (disabled)
        :param show_edges: Whether to draw triangle edges. Default True
        :param finalize: Whether to call pyplot.show() after constructing the plot. Default True
        """
        from matplotlib import pyplot
        import skimage.measure
        # Claims to be unused, but needed for subplot(projection='3d')
        from mpl_toolkits.mplot3d import Axes3D

        # Get data from self.grids
        grid = self.grids[which_shifts][::sample_period, ::sample_period, ::sample_period]
        if level is None:
            level = grid.mean()

        # Find isosurface with marching cubes
        verts, faces = skimage.measure.marching_cubes(grid, level)

        # Convert vertices from index to position
        pos_verts = numpy.array([self.ind2pos(verts[i, :], which_shifts, round_ind=False)
                                 for i in range(verts.shape[0])], dtype=float)
        xs, ys, zs = (pos_verts[:, a] for a in range(3))

        # Draw the plot
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        if show_edges:
            ax.plot_trisurf(xs, ys, faces, zs)
        else:
            ax.plot_trisurf(xs, ys, faces, zs, edgecolor='none')

        # Add a fake plot of a cube to force the axes to be equal lengths
        max_range = numpy.array([xs.max() - xs.min(),
                                 ys.max() - ys.min(),
                                 zs.max() - zs.min()], dtype=float).max()
        mg = numpy.mgrid[-1:2:2, -1:2:2, -1:2:2]
        xbs = 0.5 * max_range * mg[0].flatten() + 0.5 * (xs.max() + xs.min())
        ybs = 0.5 * max_range * mg[1].flatten() + 0.5 * (ys.max() + ys.min())
        zbs = 0.5 * max_range * mg[2].flatten() + 0.5 * (zs.max() + zs.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(xbs, ybs, zbs):
            ax.plot([xb], [yb], [zb], 'w')

        if finalize:
            pyplot.show()
