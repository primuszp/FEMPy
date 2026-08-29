"""Több terhelési eset megoldása egyetlen ritka LU-faktorizációval."""

from primfem import LinearElasticMaterial, Model, rectangular_quad_mesh

mesh = rectangular_quad_mesh(nx=30, ny=6, width=300.0, height=60.0)
model = Model(
    mesh,
    LinearElasticMaterial(young_modulus=210_000.0, poisson_ratio=0.3),
    thickness=5.0,
    name="cantilever load cases",
)

# A közös támasz minden később létrehozott esetbe bekerül.
model.fix_boundary("left")

vertical = model.load_case("vertical")
vertical.add_boundary_force("right", fy=-1000.0)

horizontal = model.load_case("horizontal")
horizontal.add_boundary_force("right", fx=1000.0)

results = model.solve_cases((vertical, horizontal), reuse_factorization=True)
for name, result in results.items():
    print(result.summary())
    print(f"  factorization reused: {result.solver_info.factorization_reused}")
    result.write(f"{name}.vtu")
