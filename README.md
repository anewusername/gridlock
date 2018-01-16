# Gridlock README

Gridlock is a Python module for drawing on coupled grids.

Gridlock is used primarily for 'painting' shapes in 3D on multiple grids which represent the
same spatial region, but are offset from each other. It does straightforward natural <-> grid unit
conversion and can handle non-uniform rectangular grids (the entire grid is generated based on
the coordinates of the boundary points along each axis).

## Installation

Requirements:
* python 3 (written and tested with 3.5)
* numpy
* [float_raster](https://mpxd.net/code/jan/float_raster)
* matplotlib (optional, used for visualization functions)
* mpl_toolkits.mplot3d (optional, used for isosurface visualization)
* skimage (optional, used for isosurface visualization)

Install with pip, via git:

```bash
pip install git+https://mpxd.net/code/jan/float_raster.git@release
pip install git+https://mpxd.net/code/jan/gridlock.git@release
```
