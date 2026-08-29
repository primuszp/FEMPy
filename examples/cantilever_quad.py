"""Befogott lemez strukturált Quad4 hálóval – minimális oktatópélda.

A fájl bemutatja az API öt fő lépését: háló, anyag, modell, peremfeltételek,
megoldás/utófeldolgozás. A használt egységrendszer N–mm–MPa.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from primfem import LinearElasticMaterial, Model, PlaneCondition, rectangular_quad_mesh

# 20×5 elem, vagyis 21×6 csomópont a 200×50 mm-es tartományon.
mesh = rectangular_quad_mesh(nx=20, ny=5, width=200.0, height=50.0)

# A Young-modulus N/mm² = MPa egységű. A sűrűség most nem szükséges, mert
# nincs testsúlyterhelés.
steel = LinearElasticMaterial(young_modulus=210_000.0, poisson_ratio=0.3, name="steel")

# A lemez 5 mm vastag. Az alapértelmezett feltételezés plane stress.
model = Model(mesh, steel, thickness=5.0, condition=PlaneCondition.STRESS, name="cantilever")

# A strukturált háló automatikusan elnevezi mind a négy peremet.
model.fix_boundary("left")

# A teljes eredőt adjuk meg; az API hálófüggetlenül, konzisztensen osztja el.
model.add_boundary_force("right", fy=-1_000.0)

# A solve csak a redukált ritka Kff mátrixot építi fel.
result = model.solve()
print(result.summary())

# A VTK a ParaView-hoz, a Matplotlib-ábra gyors Pythonos ellenőrzéshez készül.
result.write_vtk(Path(__file__).with_name("cantilever.vtk"))
result.plot(scale=100.0)
plt.show()
