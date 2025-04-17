from typing import Protocol, TypedDict, runtime_checkable, cast
from dataclasses import dataclass


class GridError(Exception):
    """ Base error type for `gridlock` """
    pass


class ExtentDict(TypedDict, total=False):
    """
    Geometrical definition of an extent (1D bounded region)
    Must contain exactly two of `min`, `max`, `center`, or `span`.
    """
    min: float
    center: float
    max: float
    span: float


@runtime_checkable
class ExtentProtocol(Protocol):
    """
    Anything that looks like an `Extent`
    """
    center: float
    span: float

    @property
    def max(self) -> float: ...

    @property
    def min(self) -> float: ...


@dataclass(init=False, slots=True)
class Extent(ExtentProtocol):
    """
    Geometrical definition of an extent (1D bounded region)
    May be constructed with any two of `min`, `max`, `center`, or `span`.
    """
    center: float
    span: float

    @property
    def max(self) -> float:
        return self.center + self.span / 2

    @property
    def min(self) -> float:
        return self.center - self.span / 2

    def __init__(
            self,
            *,
            min: float | None = None,
            center: float | None = None,
            max: float | None = None,
            span: float | None = None,
            ) -> None:
        if sum(cc is None for cc in (min, center, max, span)) != 2:
            raise GridError('Exactly two of min, center, max, span must be None!')

        if span is None:
            if center is None:
                assert min is not None
                assert max is not None
                assert max >= min
                center = 0.5 * (max + min)
                span = max - min
            elif max is None:
                assert min is not None
                assert center is not None
                span = 2 * (center - min)
            elif min is None:
                assert center is not None
                assert max is not None
                span = 2 * (max - center)
        else:                        # noqa: PLR5501
            if center is not None:
                pass
            elif max is None:
                assert min is not None
                assert span is not None
                center = min + 0.5 * span
            elif min is None:
                assert max is not None
                assert span is not None
                center = max - 0.5 * span

        assert center is not None
        assert span is not None
        if hasattr(center, '__len__'):
            assert len(center) == 1
        if hasattr(span, '__len__'):
            assert len(span) == 1
        self.center = center
        self.span = span


class SlabDict(TypedDict, total=False):
    """
    Geometrical definition of a slab (3D region bounded on one axis only)
    Must contain `axis` plus any two of `min`, `max`, `center`, or `span`.
    """
    min: float
    center: float
    max: float
    span: float
    axis: int | str


@runtime_checkable
class SlabProtocol(ExtentProtocol, Protocol):
    """
    Anything that looks like a `Slab`
    """
    axis: int
    center: float
    span: float

    @property
    def max(self) -> float: ...

    @property
    def min(self) -> float: ...


@dataclass(init=False, slots=True)
class Slab(Extent, SlabProtocol):
    """
    Geometrical definition of a slab (3D region bounded on one axis only)
    May be constructed with `axis` (bounded axis) plus any two of `min`, `max`, `center`, or `span`.
    """
    axis: int

    def __init__(
            self,
            axis: int | str,
            *,
            min: float | None = None,
            center: float | None = None,
            max: float | None = None,
            span: float | None = None,
            ) -> None:
        Extent.__init__(self, min=min, center=center, max=max, span=span)

        if isinstance(axis, str):
            axis_int = 'xyz'.find(axis.lower())
        else:
            axis_int = axis
        if axis_int not in range(3):
            raise GridError(f'Invalid axis (slab normal direction): {axis}')
        self.axis = axis_int

    def as_plane(self, where: str) -> 'Plane':
        if where == 'center':
            return Plane(axis=self.axis, pos=self.center)
        if where == 'min':
            return Plane(axis=self.axis, pos=self.min)
        if where == 'max':
            return Plane(axis=self.axis, pos=self.max)
        raise GridError(f'Invalid {where=}')


class PlaneDict(TypedDict, total=False):
    """
    Geometrical definition of a plane (2D unbounded region in 3D space)
    Must contain exactly one of `x`, `y`, `z`, or both `axis` and `pos`
    """
    x: float
    y: float
    z: float
    axis: int
    pos: float


@runtime_checkable
class PlaneProtocol(Protocol):
    """
    Anything that looks like a `Plane`
    """
    axis: int
    pos: float


@dataclass(init=False, slots=True)
class Plane(PlaneProtocol):
    """
    Geometrical definition of a plane (2D unbounded region in 3D space)
    May be constructed with any of `x=4`, `y=5`, `z=-5`, or `axis=2, pos=-5`.
    """
    axis: int
    pos: float

    def __init__(
            self,
            *,
            axis: int | str | None = None,
            pos: float | None = None,
            x: float | None = None,
            y: float | None = None,
            z: float | None = None,
            ) -> None:
        xx = x
        yy = y
        zz = z

        if sum(aa is not None for aa in (pos, xx, yy, zz)) != 1:
            raise GridError('Exactly one of pos, x, y, z must be non-None!')
        if (axis is None) != (pos is None):
            raise GridError('Either both or neither of `axis` and `pos` must be defined.')

        if isinstance(axis, str):
            axis_int = 'xyz'.find(axis.lower())
        elif axis is None:
            axis_int = (xx is None, yy is None, zz is None).index(False)
        else:
            axis_int = axis

        if axis_int not in range(3):
            raise GridError(f'Invalid axis (slab normal direction): {axis=} {x=} {y=} {z=}')
        self.axis = axis_int

        if pos is not None:
            cpos = pos
        else:
            cpos = cast('float', (xx, yy, zz)[axis_int])
            assert cpos is not None

        if hasattr(cpos, '__len__'):
            assert len(cpos) == 1
        self.pos = cpos

