"""Six coloured views of the same cantilever analysis."""

from pathlib import Path

import matplotlib.pyplot as plt

from fempy import LinearElasticMaterial, Model, rectangular_quad_mesh

mesh = rectangular_quad_mesh(nx=24, ny=6, width=240.0, height=60.0)
material = LinearElasticMaterial(young_modulus=210_000.0, poisson_ratio=0.3)
model = Model(mesh, material, thickness=4.0, name="coloured cantilever")
model.fix_nodes(mesh.nodes_where(x=0.0))

loaded_nodes = mesh.nodes_where(x=240.0)
model.add_nodal_loads(loaded_nodes, fy=-2_000.0 / len(loaded_nodes))
result = model.solve()

figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
views = (
    ("displacement_magnitude", "magma"),
    ("nodal_von_mises", "turbo"),
    ("nodal_stress_x", "coolwarm"),
    ("nodal_stress_y", "Spectral_r"),
    ("nodal_stress_xy", "seismic"),
)
for ax, (field, colour_map) in zip(axes.flat[:-1], views, strict=True):
    result.plot(scale=80.0, field=field, cmap=colour_map, ax=ax)

result.plot_principal_directions(scale=80.0, stride=2, ax=axes.flat[-1])
figure.suptitle("FEMPy Edu — coloured result gallery", fontsize=16)
output = Path(__file__).with_name("colored_cantilever.png")
figure.savefig(output, dpi=180)
print(result.summary())
print(f"Saved: {output}")
plt.show()
