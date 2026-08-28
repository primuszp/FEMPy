# FEMPy API – gyors referencia

Ez az oldal a mindennapi használathoz szükséges publikus fogalmakat mutatja.
A belső numerikus lépések részletes leírása a [KODMAGYARAZAT.md](KODMAGYARAZAT.md)
dokumentumban található.

## Az alapfolyamat

```text
Geometry2D ──GmshMesher──▶ Mesh
                              │
Mesh + Material ───────────▶ Model ──solve()──▶ AnalysisResult
```

Strukturált feladatnál a `Geometry2D` és `GmshMesher` helyett közvetlenül a
`rectangular_quad_mesh()`, `rectangular_tri_mesh()` vagy
`rectangular_t6_mesh()` használható.

## Geometria és háló

```python
geometry = (
    Geometry2D("plate")
    .add_rectangle(100.0, 50.0)
    .add_circle((50.0, 25.0), 8.0, boundary="hole", mesh_size=1.0)
)

mesh = GmshMesher(element_size=4.0, element_shape="quad").generate(geometry)
```

| Hívás | Jelentés |
|---|---|
| `add_polygon(points, boundary_names=...)` | általános egyenes oldalú kontúr |
| `add_circle(center, radius, boundary=...)` | kör alakú külső kontúr vagy lyuk |
| `add_loop(segments)` | egyenesekből és körívekből álló kontúr |
| `set_boundary_size(name, size)` | helyi hálófinomítás |
| `GmshMesher(size, "triangle")` | Triangle3 háló |
| `GmshMesher(size, "triangle", order=2)` | másodrendű Triangle6 (T6) háló |
| `GmshMesher(size, "quad")` | rekombinált Quad4 vagy vegyes háló |
| `to_quadratic_tri_mesh(mesh)` | meglévő Triangle3 háló kompatibilis T6 hálóvá emelése |

Lekérdezés és ábrázolás:

```python
mesh.node_count
mesh.element_count
mesh.boundary_names
mesh.boundary_nodes("left")
mesh.boundary_edges("hole")
mesh.plot()
mesh.plot_boundaries()
```

## Anyag és modell

```python
steel = LinearElasticMaterial(
    young_modulus=210_000.0,
    poisson_ratio=0.3,
    density=7.85e-9,
    name="steel",
)

model = Model(
    mesh,
    steel,
    thickness=5.0,
    condition="plane_stress",
    name="plate",
)
```

A könyvtár nem választ mértékegységet. Minden adatot ugyanabban a konzisztens
rendszerben kell megadni.

## Megtámasztások

```python
model.fix_node(0)                              # ux=uy=0
model.fix_boundary("left", x=True, y=False)   # csak ux=0
model.fix_boundary("bottom", x=False, y=True) # csak uy=0
model.prescribe_boundary("right", ux=0.05)    # előírt elmozdulás
```

Az egymást metsző peremek előírásai összeadódnak. Egy x-ben rögzített és egy
y-ban rögzített perem közös sarka mindkét irányban kötött lesz.

## Terhek

```python
model.add_nodal_load(node=10, fx=100.0, fy=-50.0)
model.add_boundary_traction("right", tx=2.0, ty=-1.0)
model.add_boundary_pressure("hole", pressure=10.0)
model.set_body_acceleration(ay=-9.81)
```

- A csomóponti terhelés erő dimenziójú.
- A traction és a nyomás erő/felület dimenziójú; a modell vastagsága része az
  integrálásnak.
- A pozitív nyomás a test belseje felé hat, furatnál tehát a furatból az anyag
  felé mutat.
- A testsúlyhoz az anyag sűrűségét is meg kell adni.

```python
model.plot_boundary_conditions()
```

## Megoldás

```python
result = model.solve()          # automatikus solver
result = model.solve("direct") # ritka direkt
result = model.solve("cg")     # Jacobi-CG
```

Nagy feladathoz:

```python
result = model.solve(
    SolverOptions(
        method="cg",
        relative_tolerance=1e-10,
        max_iterations=20_000,
        preconditioner="jacobi",
    )
)
```

## Eredmények

```python
result.displacement
result.reaction
result.strain
result.stress
result.von_mises
result.nodal_stress
result.nodal_von_mises
result.principal_stress
result.integration_point_stress
result.solver_info
```

```python
print(result.summary())
style = PlotStyle(language="hu", length_unit="mm", stress_unit="MPa")
result.plot(
    field="nodal_von_mises",
    scale=result.suggested_deformation_scale(),
    style=style,
)
result.plot_principal_directions()
result.write_vtk("result.vtk")
```

`PlotStyle(language="hu")` minden lebegőpontos skálán tizedesvesszőt,
`PlotStyle(language="en")` tizedespontot használ. A feszültségmezők görög
szimbólumot, megadott mértékegységet és automatikus mérnöki kitevőt kapnak.

## Diagnosztika és validáció

```python
model.plot_stiffness_matrix(kind="sparsity", reduced=True)
model.plot_stiffness_matrix(kind="magnitude")

report = run_classic_validations()
print(report.summary())
assert report.passed
```

## Hibák értelmezése

| Hiba | Tipikus ok |
|---|---|
| `stiffness matrix is singular` | hiányos megtámasztás vagy leváló hálórész |
| `inverted or degenerate Jacobian` | hibás csomópontsorrend vagy torz elem |
| `unknown boundary` | elírt vagy a geometriában nem definiált peremnév |
| `CG did not converge` | rosszul kondicionált modell vagy túl szigorú beállítás |
| `GmshNotInstalledError` | telepítsd a `.[gmsh]` opcionális függőséget |
