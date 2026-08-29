"""Belső segédfüggvények lineáris és kvadratikus peremélekhez.

A peremgeometria egy helyen kezeli a T3/Q4 kétcsomópontos éleit és a T6
háromcsomópontos, akár görbült éleit. Ugyanezt a kvadratúrát használja a
peremhossz és a konzisztens peremterhelés, ezért a két számítás nem tud
eltérő geometriai feltételezésre kerülni.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .elements import Triangle6


def element_boundary_edges(element) -> tuple[tuple[int, ...], ...]:
    """Az elem topológiai oldalai, T6 esetén a középcsomóponttal."""

    ids = element.node_ids
    if isinstance(element, Triangle6):
        return (
            (ids[0], ids[3], ids[1]),
            (ids[1], ids[4], ids[2]),
            (ids[2], ids[5], ids[0]),
        )
    return tuple((ids[index], ids[(index + 1) % len(ids)]) for index in range(len(ids)))


def edge_quadrature(
    edge: tuple[int, ...],
    nodes: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Lineáris vagy kvadratikus él integrációs adatait adja.

    Minden rekord ``(hely, érintő, alakfüggvény, súly)``. A T6 élen
    hárompontos Gauss-szabály integrálja a kvadratikus geometriát.
    """

    coordinates = nodes[list(edge)]
    if len(edge) == 2:
        points = ((0.0, 2.0),)
    else:
        gauss = np.sqrt(3.0 / 5.0)
        points = ((-gauss, 5.0 / 9.0), (0.0, 8.0 / 9.0), (gauss, 5.0 / 9.0))
    for coordinate, weight in points:
        if len(edge) == 2:
            shape = np.array((0.5 * (1.0 - coordinate), 0.5 * (1.0 + coordinate)))
            derivative = np.array((-0.5, 0.5))
        else:
            shape = np.array(
                (
                    0.5 * coordinate * (coordinate - 1.0),
                    1.0 - coordinate**2,
                    0.5 * coordinate * (coordinate + 1.0),
                )
            )
            derivative = np.array((coordinate - 0.5, -2.0 * coordinate, coordinate + 0.5))
        yield shape @ coordinates, derivative @ coordinates, shape, weight


def edge_length(edge: tuple[int, ...], nodes: np.ndarray) -> float:
    """Egy lineáris vagy kvadratikus peremél Gauss-integrált hossza."""

    return float(
        sum(
            np.linalg.norm(tangent) * weight
            for _, tangent, _, weight in edge_quadrature(edge, nodes)
        )
    )
