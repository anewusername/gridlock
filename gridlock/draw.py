"""
Drawing-related methods for Grid class
"""
from typing import List, Optional, Union, Sequence, Callable

import numpy        # type: ignore
from numpy import diff, floor, ceil, zeros, hstack, newaxis

from float_raster import raster

from . import GridError, Direction
from ._helpers import is_scalar


eps_callable_t = Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray], numpy.ndarray]


def draw_polygons(self,
                  surface_normal: Union[Direction, int],
                  center: numpy.ndarray,
                  polygons: Sequence[numpy.ndarray],
                  thickness: float,
                  eps: Union[Sequence[Union[float, eps_callable_t]], float, eps_callable_t],
                  ) -> None:
    """
    Draw polygons on an axis-aligned plane.

    Args:
        surface_normal: Axis normal to the plane we're drawing on. Can be a `Direction` or
            integer in `range(3)`
        center: 3-element ndarray or list specifying an offset applied to all the polygons
        polygons: List of Nx2 or Nx3 ndarrays, each specifying the vertices of a polygon
            (non-closed, clockwise). If Nx3, the surface_normal coordinate is ignored. Each
            polygon must have at least 3 vertices.
        thickness: Thickness of the layer to draw
        eps: Value to draw with ('epsilon'). Can be scalar, callable, or a list
            of any of these (1 per grid). Callable values should take an ndarray the shape of the
            grid and return an ndarray of equal shape containing the eps value at the given x, y,
            and z (natural, not grid coordinates).

    Raises:
        GridError
    """
    # Turn surface_normal into its integer representation
    if isinstance(surface_normal, Direction):
        surface_normal = surface_normal.value
    assert(isinstance(surface_normal, int))

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

    # ## Generate weighing function
    def to_3d(vector: numpy.ndarray, val: float = 0.0) -> numpy.ndarray:
        v_2d = numpy.array(vector, dtype=float)
        return numpy.insert(v_2d, surface_normal, (val,))

    # iterate over grids
    for i, grid in enumerate(self.grids):
        # ## Evaluate or expand eps[i]
        if callable(eps[i]):
            # meshgrid over the (shifted) domain
            domain = [self.shifted_xyz(i)[k][bdi_min[k]:bdi_max[k]+1] for k in range(3)]
            (x0, y0, z0) = numpy.meshgrid(*domain, indexing='ij')

            # evaluate on the meshgrid
            eps_i = eps[i](x0, y0, z0)
            if not numpy.isfinite(eps_i).all():
                raise GridError('Non-finite values in eps[%u]' % i)
        elif not is_scalar(eps[i]):
            raise GridError('Unsupported eps[{}]: {}'.format(i, type(eps[i])))
        else:
            # eps[i] is scalar non-callable
            eps_i = eps[i]

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
            centers_slice = tuple(numpy.s_[i:f + 1] for i, f in zip(corner_min - bdi_min[surface],
                                                                    corner_max - bdi_min[surface]))

            aa_x, aa_y = (self.shifted_exyz(i)[a][s] for a, s in zip(surface, edge_slices))
            w_xy[centers_slice] += raster(polygon.T, aa_x, aa_y)

        # Clamp overlapping polygons to 1
        w_xy = numpy.minimum(w_xy, 1.0)

        # 2) Generate weights in z-direction
        w_z = numpy.zeros(((bdi_max - bdi_min + 1)[surface_normal], ))

        def get_zi(offset, i=i, w_z=w_z):
            edges = self.shifted_exyz(i)[surface_normal]
            point = center[surface_normal] + offset
            grid_coord = numpy.digitize(point, edges) - 1
            w_coord = grid_coord - bdi_min[surface_normal]

            if w_coord < 0:
                w_coord = 0
                f = 0
            elif w_coord >= w_z.size:
                w_coord = w_z.size - 1
                f = 1
            else:
                dz = self.shifted_dxyz(i)[surface_normal][grid_coord]
                f = (point - edges[grid_coord]) / dz
            return f, w_coord

        zi_top_f, zi_top = get_zi(+thickness / 2.0)
        zi_bot_f, zi_bot = get_zi(-thickness / 2.0)

        w_z[zi_bot + 1:zi_top] = 1

        if zi_bot < zi_top:
            w_z[zi_top] = zi_top_f
            w_z[zi_bot] = 1 - zi_bot_f
        else:
            w_z[zi_bot] = zi_top_f - zi_bot_f

        # 3) Generate total weight function
        w = (w_xy[:, :, newaxis] * w_z).transpose(numpy.insert([0, 1], surface_normal, (2,)))

        # ## Modify the grid
        g_slice = (i,) + tuple(numpy.s_[bdi_min[a]:bdi_max[a] + 1] for a in range(3))
        self.grids[g_slice] = (1 - w) * self.grids[g_slice] + w * eps_i


def draw_polygon(self,
                 surface_normal: Union[Direction, int],
                 center: numpy.ndarray,
                 polygon: numpy.ndarray,
                 thickness: float,
                 eps: Union[Sequence[Union[float, eps_callable_t]], float, eps_callable_t],
                 ) -> None:
    """
    Draw a polygon on an axis-aligned plane.

    Args:
        surface_normal: Axis normal to the plane we're drawing on. Can be a Direction or
            integer in range(3)
        center: 3-element ndarray or list specifying an offset applied to the polygon
        polygon: Nx2 or Nx3 ndarray specifying the vertices of a polygon (non-closed,
            clockwise). If Nx3, the surface_normal coordinate is ignored. Must have at
            least 3 vertices.
        thickness: Thickness of the layer to draw
        eps: Value to draw with ('epsilon'). See `draw_polygons()` for details.
    """
    self.draw_polygons(surface_normal, center, [polygon], thickness, eps)


def draw_slab(self,
              surface_normal: Union[Direction, int],
              center: numpy.ndarray,
              thickness: float,
              eps: Union[List[Union[float, eps_callable_t]], float, eps_callable_t],
              ) -> None:
    """
    Draw an axis-aligned infinite slab.

    Args:
        surface_normal: Axis normal to the plane we're drawing on. Can be a `Direction` or
            integer in `range(3)`
        center: Surface_normal coordinate at the center of the slab
        thickness: Thickness of the layer to draw
        eps: Value to draw with ('epsilon'). See `draw_polygons()` for details.
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


def draw_cuboid(self,
                center: numpy.ndarray,
                dimensions: numpy.ndarray,
                eps: Union[List[Union[float, eps_callable_t]], float, eps_callable_t],
                ) -> None:
    """
    Draw an axis-aligned cuboid

    Args:
        center: 3-element ndarray or list specifying the cuboid's center
        dimensions: 3-element list or ndarray containing the x, y, and z edge-to-edge
             sizes of the cuboid
        eps: Value to draw with ('epsilon'). See `draw_polygons()` for details.
    """
    p = numpy.array([[-dimensions[0], +dimensions[1]],
                     [+dimensions[0], +dimensions[1]],
                     [+dimensions[0], -dimensions[1]],
                     [-dimensions[0], -dimensions[1]]], dtype=float) / 2.0
    thickness = dimensions[2]
    self.draw_polygon(Direction.z, center, p, thickness, eps)


def draw_cylinder(self,
                  surface_normal: Union[Direction, int],
                  center: numpy.ndarray,
                  radius: float,
                  thickness: float,
                  num_points: int,
                  eps: Union[List[Union[float, eps_callable_t]], float, eps_callable_t],
                  ) -> None:
    """
    Draw an axis-aligned cylinder. Approximated by a num_points-gon

    Args:
        surface_normal: Axis normal to the plane we're drawing on. Can be a `Direction` or
            integer in `range(3)`
        center: 3-element ndarray or list specifying the cylinder's center
        radius: cylinder radius
        thickness: Thickness of the layer to draw
        num_points: The circle is approximated by a polygon with `num_points` vertices
        eps: Value to draw with ('epsilon'). See `draw_polygons()` for details.
    """
    theta = numpy.linspace(0, 2*numpy.pi, num_points, endpoint=False)
    x = radius * numpy.sin(theta)
    y = radius * numpy.cos(theta)
    polygon = hstack((x[:, newaxis], y[:, newaxis]))
    self.draw_polygon(surface_normal, center, polygon, thickness, eps)


def draw_extrude_rectangle(self,
                           rectangle: numpy.ndarray,
                           direction: Union[Direction, int],
                           polarity: int,
                           distance: float,
                           ) -> None:
    """
    Extrude a rectangle of a previously-drawn structure along an axis.

    Args:
        rectangle: 2x3 ndarray or list specifying the rectangle's corners
        direction: Direction to extrude in. Direction enum or int in range(3)
        polarity: +1 or -1, direction along axis to extrude in
        distance: How far to extrude
    """
    # Turn extrude_direction into its integer representation
    if isinstance(direction, Direction):
        direction = direction.value
    assert(isinstance(direction, int))

    s = numpy.sign(polarity)

    rectangle = numpy.array(rectangle, dtype=float)
    if s == 0:
        raise GridError('0 is not a valid polarity')
    if direction not in range(3):
        raise GridError('Invalid direction: {}'.format(direction))
    if rectangle[0, direction] != rectangle[1, direction]:
        raise GridError('Rectangle entries along extrusion direction do not match.')

    center = rectangle.sum(axis=0) / 2.0
    center[direction] += s * distance / 2.0

    surface = numpy.delete(range(3), direction)

    dim = numpy.fabs(diff(rectangle, axis=0).T)[surface]
    p = numpy.vstack((numpy.array([-1, -1, 1, 1], dtype=float) * dim[0]/2.0,
                      numpy.array([-1, 1, 1, -1], dtype=float) * dim[1]/2.0)).T
    thickness = distance

    eps_func = []
    for i, grid in enumerate(self.grids):
        z = self.pos2ind(rectangle[0, :], i, round_ind=False, check_bounds=False)[direction]

        ind = [int(floor(z)) if i == direction else slice(None) for i in range(3)]

        fpart = z - floor(z)
        mult = [1-fpart, fpart][::s]  # reverses if s negative

        eps = mult[0] * grid[tuple(ind)]
        ind[direction] += 1
        eps += mult[1] * grid[tuple(ind)]

        def f_eps(xs, ys, zs, i=i, eps=eps) -> numpy.ndarray:
            # transform from natural position to index
            xyzi = numpy.array([self.pos2ind(qrs, which_shifts=i)
                                for qrs in zip(xs.flat, ys.flat, zs.flat)], dtype=int)
            # reshape to original shape and keep only in-plane components
            qi, ri = (numpy.reshape(xyzi[:, k], xs.shape) for k in surface)
            return eps[qi, ri]

        eps_func.append(f_eps)

    self.draw_polygon(direction, center, p, thickness, eps_func)

