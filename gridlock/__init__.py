"""
3D coupled grid generator

Grid generator, used primarily for 'painting' shapes in 3D on multiple grids which represent the
 same spatial region, but are offset from each other. It does straightforward natural <-> grid unit
 conversion and can handle non-uniform rectangular grids (the entire grid is generated based on
 the coordinates of the boundary points along each axis).

Its primary purpose is for drawing Yee grids for electromagnetic simulations.


Dependencies:
- numpy
- matplotlib            [Grid.visualize_*]
- mpl_toolkits.mplot3d  [Grid.visualize_isosurface()]
- skimage               [Grid.visualize_isosurface()]
"""
from .error import GridError as GridError
from .grid import Grid as Grid

__author__ = 'Jan Petykiewicz'
__version__ = '1.2'
version = __version__
