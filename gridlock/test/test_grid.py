# import pytest
import numpy
from numpy.testing import assert_allclose       #, assert_array_equal

from .. import Grid, Extent  #, Slab, Plane


def test_draw_oncenter_2x2() -> None:
    xs = [-1, 0, 1]
    ys = [-1, 0, 1]
    zs = [-1, 1]
    grid = Grid([xs, ys, zs], shifts=[[0, 0, 0]])
    arr = grid.allocate(0)

    grid.draw_cuboid(
        arr,
        x=dict(center=0, span=1),
        y=Extent(center=0, span=1),
        z=dict(center=0, span=10),
        foreground=1,
        )

    correct = numpy.array([[0.25, 0.25],
                           [0.25, 0.25]])[None, :, :, None]

    assert_allclose(arr, correct)


def test_draw_ongrid_4x4() -> None:
    xs = [-2, -1, 0, 1, 2]
    ys = [-2, -1, 0, 1, 2]
    zs = [-1, 1]
    grid = Grid([xs, ys, zs], shifts=[[0, 0, 0]])
    arr = grid.allocate(0)

    grid.draw_cuboid(
        arr,
        x=dict(center=0, span=2),
        y=dict(min=-1, max=1),
        z=dict(center=0, min=-5),
        foreground=1,
        )

    correct = numpy.array([[0, 0, 0, 0],
                           [0, 1, 1, 0],
                           [0, 1, 1, 0],
                           [0, 0, 0, 0]])[None, :, :, None]

    assert_allclose(arr, correct)


def test_draw_xshift_4x4() -> None:
    xs = [-2, -1, 0, 1, 2]
    ys = [-2, -1, 0, 1, 2]
    zs = [-1, 1]
    grid = Grid([xs, ys, zs], shifts=[[0, 0, 0]])
    arr = grid.allocate(0)

    grid.draw_cuboid(
        arr,
        x=dict(center=0.5, span=1.5),
        y=dict(min=-1, max=1),
        z=dict(center=0, span=10),
        foreground=1,
        )

    correct = numpy.array([[0,    0,    0, 0],
                           [0, 0.25, 0.25, 0],
                           [0,    1,    1, 0],
                           [0, 0.25, 0.25, 0]])[None, :, :, None]

    assert_allclose(arr, correct)


def test_draw_yshift_4x4() -> None:
    xs = [-2, -1, 0, 1, 2]
    ys = [-2, -1, 0, 1, 2]
    zs = [-1, 1]
    grid = Grid([xs, ys, zs], shifts=[[0, 0, 0]])
    arr = grid.allocate(0)

    grid.draw_cuboid(
        arr,
        x=dict(min=-1, max=1),
        y=dict(center=0.5, span=1.5),
        z=dict(center=0, span=10),
        foreground=1,
        )

    correct = numpy.array([[0,    0, 0,    0],
                           [0, 0.25, 1, 0.25],
                           [0, 0.25, 1, 0.25],
                           [0,    0, 0,    0]])[None, :, :, None]

    assert_allclose(arr, correct)


def test_draw_2shift_4x4() -> None:
    xs = [-2, -1, 0, 1, 2]
    ys = [-2, -1, 0, 1, 2]
    zs = [-1, 1]
    grid = Grid([xs, ys, zs], shifts=[[0, 0, 0]])
    arr = grid.allocate(0)

    grid.draw_cuboid(
        arr,
        x=dict(center=0.5, span=1.5),
        y=dict(min=-0.5, max=0.5),
        z=dict(center=0, span=10),
        foreground=1,
        )

    correct = numpy.array([[0,     0,     0, 0],
                           [0, 0.125, 0.125, 0],
                           [0,   0.5,   0.5, 0],
                           [0, 0.125, 0.125, 0]])[None, :, :, None]

    assert_allclose(arr, correct)
