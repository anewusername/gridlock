import numpy
from gridlock import Grid


if __name__ == '__main__':
    # xyz = [numpy.arange(-5.0, 6.0), numpy.arange(-4.0, 5.0), [-1.0, 1.0]]
    # eg = Grid(xyz)
    # egc = Grid.allocate(0.0)
    # # eg.draw_slab(egc, slab=dict(axis=2, center=0, span=10), foreground=2)
    # eg.draw_cylinder(egc, h=slab(axis=2, center=0, span=10),
    #     center2d=[0, 0], radius=4, thickness=10, num_points=1000, foreground=1)
    # eg.visualize_slice(egc, plane=dict(z=0), which_shifts=2)

    # xyz2 = [numpy.arange(-5.0, 6.0), [-1.0, 1.0], numpy.arange(-4.0, 5.0)]
    # eg2 = Grid(xyz2)
    # eg2c = Grid.allocate(0.0)
    # # eg2.draw_slab(eg2c, slab=dict(axis=2, center=0, span=10), foreground=2)
    # eg2.draw_cylinder(eg2c, h=slab(axis=1, center=0, span=10), center2d=[0, 0],
    #     radius=4, num_points=1000, foreground=1.0)
    # eg2.visualize_slice(eg2c, plane=dict(y=0), which_shifts=1)

    # n = 20
    # m = 3
    # r1 = numpy.fromfunction(lambda x: numpy.sign(x - n) * 2 ** (abs(x - n)/m), (2*n, ))
    # print(r1)
    # xyz3 = [r1, numpy.linspace(-5.5, 5.5, 30), numpy.linspace(-5.5, 5.5, 10)]
    # xyz3 = [numpy.linspace(-5.5, 5.5, 10),
    #         numpy.linspace(-5.5, 5.5, 10),
    #         numpy.linspace(-5.5, 5.5, 10)]

    half_x = [.25, .5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5]
    xyz3 = [numpy.array([-x for x in half_x[::-1]] + [0] + half_x, dtype=float),
            numpy.linspace(-5.5, 5.5, 10, dtype=float),
            numpy.linspace(-5.5, 5.5, 10, dtype=float)]
    eg = Grid(xyz3)
    egc = eg.allocate(0)
    # eg.draw_slab(Direction.z, 0, 10, 2)
    eg.save('/home/jan/Desktop/test.pickle')
    eg.draw_cylinder(
        egc,
        h=dict(axis='z', center=0, span=10),
        center2d=[0, 0],
        radius=2.0,
        num_points=1000,
        foreground=1,
        )
    eg.draw_extrude_rectangle(
        egc,
        rectangle=[[-2, 1, -1], [0, 1, 1]],
        direction=1,
        polarity=+1,
        distance=5,
        )
    eg.visualize_slice(egc, plane=dict(z=0), which_shifts=2)
    eg.visualize_isosurface(egc, which_shifts=2)
