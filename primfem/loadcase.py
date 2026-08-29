"""Terhelési esetek egy közös végeselemes modellen.

A :class:`LoadCase` közvetlenül a :class:`primfem.model.Model` elemzési API-ját
használja. Nincs kézzel karbantartott továbbító réteg: minden új terhelési vagy
peremfeltétel-metódus automatikusan mindkét objektumon azonos.
"""

from __future__ import annotations

from .model import Model


class LoadCase(Model):
    """Egy modell önálló terhelés- és peremfeltétel-készlete.

    A példányt a :meth:`Model.load_case` készíti. A hálót és az anyagot
    megosztja a szülőmodellel, a terheket, előírt elmozdulásokat és
    testgyorsulást viszont saját tömbökben tartja.

    Az ``inherit=True`` alapérték a szülőmodell aktuális terhelési állapotát
    másolja kezdőértékként. A későbbi módosítások függetlenek maradnak.
    """

    __slots__ = ("model",)

    def __init__(self, model: Model, name: str, *, inherit: bool = True) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("load-case name must be a non-empty string")
        self.model = model
        super().__init__(
            model.mesh,
            model.material,
            thickness=model.thickness,
            condition=model.condition,
            name=name.strip(),
        )
        if inherit:
            self._loads[:] = model._loads
            self._prescribed.update(model._prescribed)
            self._body_acceleration[:] = model._body_acceleration
