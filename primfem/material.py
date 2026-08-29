"""Lineárisan rugalmas, izotróp anyagmodell.

Ebben a modulban található a kétdimenziós kontinuum-elemek Hooke-törvénye.
A végeselemes egyenletekben az anyagtörvény a következő alakban szerepel::

    sigma = D @ epsilon

ahol ``epsilon = [epsilon_x, epsilon_y, gamma_xy]``, ``sigma`` pedig
``[sigma_x, sigma_y, tau_xy]``. A :class:`PlaneCondition` választja ki, hogy
a ``D`` mátrix síkfeszültségi vagy síkalakváltozási feltételezéshez készüljön.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class PlaneCondition(str, Enum):
    """A kétdimenziós modell harmadik irányra vonatkozó feltételezése.

    ``STRESS``: vékony lemez, amelynél a síkra merőleges feszültség nulla.
    ``STRAIN``: hosszú test keresztmetszete, amelynél a síkra merőleges
    alakváltozás nulla.
    """

    STRESS = "plane_stress"
    STRAIN = "plane_strain"


@dataclass(frozen=True, slots=True)
class LinearElasticMaterial:
    """Izotróp, lineárisan rugalmas anyag.

    A csomag nem rögzít mértékegységet: minden bemenő adatnak ugyanahhoz a
    konzisztens rendszerhez kell tartoznia. Ha a koordináták mm-ben, az erők
    N-ban vannak, akkor a Young-modulust N/mm²-ben kell megadni.

    Args:
        young_modulus: Young-modulus, mindig pozitív.
        poisson_ratio: Poisson-tényező, ``-1 < nu < 0.5`` tartományban.
        density: Tömegsűrűség. Csak testsúly- és tömegmátrix-számításnál kell.
        name: Ember számára olvasható anyagnév.
    """

    young_modulus: float
    poisson_ratio: float
    density: float = 0.0
    name: str = "material"

    def __post_init__(self) -> None:
        if not np.isfinite(self.young_modulus) or self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be positive and finite")
        if not np.isfinite(self.poisson_ratio) or not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be finite and between -1 and 0.5")
        if not np.isfinite(self.density) or self.density < 0.0:
            raise ValueError("density must be finite and cannot be negative")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("material name must be a non-empty string")

    def constitutive_matrix(self, condition: PlaneCondition | str) -> NDArray[np.float64]:
        """Elkészíti a 3×3-as ``D`` anyagmátrixot.

        Args:
            condition: ``plane_stress`` vagy ``plane_strain``.

        Returns:
            A ``[sigma_x, sigma_y, tau_xy] = D @ [ex, ey, gamma_xy]``
            összefüggés NumPy-mátrixa.
        """

        condition = PlaneCondition(condition)
        e = self.young_modulus
        nu = self.poisson_ratio
        if condition is PlaneCondition.STRESS:
            # Vékony lemez: sigma_z = 0. A nyírási komponens mérnöki
            # gamma_xy alakváltozást használ, ezért D[2,2] = G.
            factor = e / (1.0 - nu**2)
            return factor * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])

        # Hosszú test keresztmetszete: epsilon_z = 0. A 0.5-höz közeli
        # Poisson-tényező közel összenyomhatatlan, ezért a mátrix rosszul
        # kondicionálttá válhat; a bemeneti ellenőrzés a pontos 0.5-öt tiltja.
        factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return factor * np.array(
            [
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
            ]
        )

    def von_mises(self, stress: NDArray[np.float64], condition: PlaneCondition | str) -> float:
        """Von Mises-egyenértékfeszültséget számít.

        Síkfeszültségnél ``sigma_z=0``. Síkalakváltozásnál a hiányzó
        ``sigma_z`` komponenst a Hooke-törvényből állítjuk vissza, így az
        egyenértékfeszültség a teljes háromdimenziós feszültségállapotot veszi
        figyelembe.

        Args:
            stress: ``[sigma_x, sigma_y, tau_xy]``.
            condition: A modell síkbeli feltételezése.

        Returns:
            Skalár von Mises-feszültség.
        """

        sx, sy, txy = np.asarray(stress, dtype=float)
        condition = PlaneCondition(condition)
        sz = 0.0 if condition is PlaneCondition.STRESS else self.poisson_ratio * (sx + sy)
        return float(
            np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * txy**2)
        )
