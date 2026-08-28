# FEMPy

[![CI](https://github.com/primuszp/FEMPy/actions/workflows/ci.yml/badge.svg)](https://github.com/primuszp/FEMPy/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Olvasható, validált és memóriahatékony kétdimenziós végeselemes könyvtár
Pythonhoz. A FEMPy célja, hogy a végeselemes módszer tanulható maradjon: a
publikus API kicsi, a numerikus lépések dokumentáltak, a globális mátrixok
ritkák, az eredmények pedig közvetlenül ábrázolhatók és exportálhatók.

A projekt a korábbi PROTUS, Pyfem és Myfem kísérletek rendezett utódja. A régi
formátumok importálhatók, de az új kód egyetlen következetes `fempy` API-t
használ.

Dokumentáció:

- [gyors API-referencia](docs/API.md);
- [részletes kódmagyarázat](docs/KODMAGYARAZAT.md);
- [klasszikus validáció](docs/VALIDACIO.md);
- [futtatható példák](examples/README.md).

## Mit tud?

- síkfeszültségi és síkalakváltozási lineáris rugalmasság;
- háromcsomópontos CST háromszög (`Triangle3`);
- hatcsomópontos kvadratikus háromszög (`Triangle6`, T6), hétpontos
  Dunavant-integrálással;
- négycsomópontos, bilineáris négyszög (`Quad4`), 2×2 Gauss-integrálással;
- strukturált négyszög- és háromszögháló generálása;
- tetszőleges sokszög és kör alakú lyukak modern Gmsh-hálózása;
- névvel ellátott peremek, helyi hálófinomítás és perem menti megoszló terhek;
- pontterhek, előírt elmozdulások és egyenletes testsúly;
- ritka globális merevségi mátrix és közvetlen egyenletmegoldás;
- reakcióerők, elemközépi alakváltozások és feszültségek;
- mind a négy Quad4 Gauss-pont eredményei és csomóponti extrapoláció;
- elemközi csomóponti átlagolás, főfeszültségek és főirányok;
- konzisztens tömegmátrix és testsúlyterhelés;
- von Mises-feszültség, Matplotlib-ábra és teljes ParaView VTK-export;
- a régi PROTUS és Myfem bemeneti fájlok beolvasása;
- automatikus klasszikus FEM-verifikáció Triangle3, Triangle6 és Quad4 elemekre.

## Telepítés

Python 3.10 vagy újabb szükséges. Telepítés GitHubról:

```powershell
python -m pip install "fempy-edu[gmsh] @ git+https://github.com/primuszp/FEMPy.git"
```

Fejlesztői telepítés a klónozott projekt gyökerében:

```powershell
python -m pip install -e .
```

A tetszőleges geometriák Gmsh-hálózásához az opcionális kiegészítés kell:

```powershell
python -m pip install -e ".[gmsh]"
```

## Az API öt lépése

Egy elemzés mindig ugyanazt az öt fogalmat követi:

1. háló létrehozása;
2. anyag megadása;
3. modell összeállítása;
4. megtámasztások és terhek felvétele;
5. megoldás és kiértékelés.

```python
from fempy import LinearElasticMaterial, Model, rectangular_quad_mesh

# 1. 200 × 50 mm-es tartomány, 20 × 5 elemmel
mesh = rectangular_quad_mesh(nx=20, ny=5, width=200.0, height=50.0)

# 2. N és mm mértékegységrendszer: E = 210 000 N/mm²
steel = LinearElasticMaterial(young_modulus=210_000.0, poisson_ratio=0.3)

# 3. 5 mm vastag síkfeszültségi modell
model = Model(mesh, steel, thickness=5.0, name="konzol")

# 4. Bal oldal befogása, jobb oldalon összesen 1000 N lefelé
model.fix_nodes(mesh.nodes_where(x=0.0))
right_edge = mesh.nodes_where(x=200.0)
model.add_nodal_loads(right_edge, fy=-1_000.0 / len(right_edge))

# 5. Megoldás
result = model.solve()
print(result.summary())
result.write_vtk("konzol.vtk")
result.plot(scale=100.0)
```

## Tetszőleges geometria modern Gmsh-hálóval

A FEMPy közvetlenül a Gmsh Python API-ját használja. Nem készít köztes
`.geo` vagy `.inp` fájlt: a geometria, a háló és a peremcímkék memóriában
haladnak át. A Gmsh opcionális, ezért a strukturált hálók és a solver nélküle
is használhatók.

```python
from fempy import Geometry2D, GmshMesher, LinearElasticMaterial, Model

geometry = (
    Geometry2D("lyukas_lemez")
    .add_rectangle(width=100.0, height=50.0)
    .add_circle(
        center=(50.0, 25.0),
        radius=8.0,
        boundary="hole",
        mesh_size=1.2,       # helyi finomítás a furatnál
    )
)

mesh = GmshMesher(
    element_size=4.0,        # globális cél-elemméret
    element_shape="triangle",
    order=2,                 # T6; order=1 esetén Triangle3
).generate(geometry)

steel = LinearElasticMaterial(210_000.0, 0.3)
model = Model(mesh, steel, thickness=5.0)
model.fix_boundary("left")
model.add_boundary_traction("right", ty=-2.0)
model.plot_boundary_conditions()
result = model.solve()

print(mesh.boundary_names)
mesh.plot()
result.plot(field="nodal_von_mises")
```

Strukturált T6 háló Gmsh nélkül is készíthető:

```python
from fempy import rectangular_t6_mesh

mesh = rectangular_t6_mesh(nx=10, ny=4, width=100.0, height=40.0)
```

Nyomás és előírt peremelmozdulás ugyanilyen név szerinti API-t használ:

```python
model.add_boundary_pressure("hole", pressure=10.0)  # pozitív: a testbe mutat
model.prescribe_boundary("right", ux=0.05)          # vezérelt elmozdulás
```

A téglalap oldalai alapból `bottom`, `right`, `top`, `left` nevet kapnak.
Általános sokszögnél a `boundary_names` paraméterrel minden oldal külön
elnevezhető. A `Mesh` a Gmsh után is megőrzi ezeket:

```python
mesh.boundary_nodes("left")
mesh.boundary_edges("hole")
mesh.plot_boundaries()                       # minden perem, névfeliratokkal
mesh.plot_boundaries(names=["left", "hole"]) # csak a kiválasztott peremek
```

Íves külső kontúrhoz az `add_loop()` metódus `LineSegment2D` és
`CircularArc2D` szegmenseket fogad. A szegmenseknek folytonos, zárt láncot
kell alkotniuk; a 180 fokos vagy nagyobb íveket több kisebb ívre kell bontani.

Az `element_shape="quad"` a Gmsh rekombinációját kapcsolja be. Egyszerű
tartományokon tiszta Quad4 hálót ad; bonyolult geometrián a Gmsh szükség esetén
vegyes Triangle3–Quad4 hálót is visszaadhat, amit a solver szintén kezel.

A teljes, futtatható és ábrát készítő példa:
`examples/gmsh_plate_with_hole.py`.

## Klasszikus FEM-validáció

A könyvtár külön validációs csomaggal ellenőrzi az elemformulát, a
peremterheléseket, a hajlítási konvergenciát és a torzított hálón mutatott
viselkedést:

```python
from fempy import run_classic_validations

report = run_classic_validations()
print(report.summary())
report.write_markdown("validation.md")
report.plot()

assert report.passed
```

A beépített esetek:

- egytengelyű patch-próba egzakt homogén elmozdulás- és feszültségmezővel;
- karcsú konzol Timoshenko-féle csúcselmozdulási referenciával;
- Cook-membrán konvergált referencia-csúcselmozdulással és öt hálószinttel;
- mindhárom feladat külön Triangle3, Triangle6 és Quad4 elemekkel.

Az elfogadás nemcsak a legfinomabb háló hibáját, hanem a hiba minden
hálófinomításnál csökkenő tendenciáját is vizsgálja. A teljes futtatás:

```powershell
python examples/validate_classic_fem.py
```

Részletes módszertan: `docs/VALIDACIO.md`. Az aktuális újragenerált eredmények:
`examples/classic_validation_results.md`.

### Színezett Matplotlib-ábrák

Az összes plot ugyanazt a `PlotStyle` objektumot használhatja. Magyar módban
a számok tizedesvesszővel, angol módban tizedesponttal jelennek meg. A
feszültségek szabványos $\sigma_x$, $\tau_{xy}$, $\sigma_1$ és
$\sigma_\mathrm{vM}$ jelölést, explicit mértékegységet és szükség esetén közös
mérnöki $10^{3n}$ skálát kapnak.

```python
from fempy import PlotStyle

magyar = PlotStyle(language="hu", length_unit="mm", stress_unit="MPa")
angol = PlotStyle(language="en", length_unit="mm", stress_unit="MPa")

result.plot(field="nodal_von_mises", style=magyar)
result.plot(field="nodal_stress_x", style=angol)
```

```python
result.plot(field="displacement_magnitude", cmap="magma", scale=100.0)
result.plot(field="nodal_von_mises", cmap="turbo", scale=100.0)
result.plot(field="nodal_stress_x", cmap="coolwarm", scale=100.0)
result.plot(field="nodal_stress_xy", cmap="seismic", scale=100.0)
result.plot(field="principal_stress_1", cmap="Spectral_r", scale=100.0)
result.plot_principal_directions(scale=100.0, stride=2)
```

A `show_undeformed=False` kikapcsolja a szaggatott eredeti hálót. A `cmap`
bármely Matplotlib-színtérkép neve lehet. A sima csomóponti mezők Gouraud-
színezést, az elemmezők elemenként állandó színezést használnak.

Két kész galéria futtatható:

```powershell
python examples/colored_cantilever.py
python examples/colored_protus.py
python examples/colored_femaster_samples.py
python examples/sparse_matrix_visualization.py
python examples/t6_localized_plots.py
```

Az `examples/femaster_samples` könyvtár a FEMaster2D hat eredeti háromszög-
elemes mintáját is tartalmazza. A galéria automatikus deformációskálát használ,
így az alakváltozott háló a modell befoglaló méretéhez igazodik. A színskála
magassága mindig pontosan a diagram rajztéglalapjának magasságát követi.

## Ritka merevségi mátrix és solver API

A normál megoldási út közvetlenül a szabad szabadságfokok redukált CSR-
mátrixát állítja össze. A teljes globális mátrix ezért nem foglal memóriát,
hacsak külön nem kérjük vizsgálathoz vagy ábrázoláshoz.

```python
from fempy import SolverOptions

# Automatikus választás: ritka direkt vagy Jacobi-CG
result = model.solve()

# Kifejezetten direkt ritka megoldó
result = model.solve("direct")

# Memóriatakarékos iteratív megoldó
result = model.solve(
    SolverOptions(
        method="cg",
        relative_tolerance=1e-10,
        max_iterations=20_000,
        preconditioner="jacobi",
    )
)

print(result.solver_info.method)
print(result.solver_info.iterations)
print(result.solver_info.relative_residual)
print(result.solver_info.matrix_memory_megabytes)
```

Az `auto` mód alapértelmezésben 20 000 szabad szabadságfokig a robusztus
SciPy ritka direkt megoldót használja. E fölött Jacobi-előkondicionált konjugált
gradiensre vált, amely nem hoz létre nagy kitöltésű faktorizációt.

### Merevségi mátrix vizualizálása

```python
import matplotlib.pyplot as plt

# Teljes vagy peremfeltételekkel redukált CSR-mátrix
K = model.stiffness_matrix()
Kff = model.stiffness_matrix(reduced=True)

# Nemnulla szerkezeti minta
model.plot_stiffness_matrix(kind="sparsity")

# A koefficiensek logaritmikus magnitúdója
model.plot_stiffness_matrix(kind="magnitude", reduced=True)
plt.show()
```

Az ábrázoló nagy mátrixot sem alakít sűrűvé. Legfeljebb `max_points` darab
nemnulla bejegyzést rajzol ki, így maga a vizualizáció memóriaigénye is
korlátozott marad.

A csomópontindexek az új Python API-ban **0-tól indulnak**. A régi importerek
automatikusan átalakítják az 1-től számozott fájlokat.

## Az eredmények elérése

```python
result.displacement          # (csomópontok, 2): ux, uy
result.displacement_magnitude
result.reaction              # (csomópontok, 2): Rx, Ry
result.strain                # (elemek, 3): ex, ey, gamma_xy
result.stress                # (elemek, 3): sx, sy, tau_xy
result.von_mises             # (elemek,)
result.principal_stress       # (elemek, 2): sigma_1, sigma_2
result.principal_angle        # (elemek,): főirány radiánban

# Gauss-pontonként elemenként eltérő hosszúságú tömbök
result.integration_point_strain
result.integration_point_stress
result.integration_point_von_mises

# Gauss-pontokból extrapolált és a közös csomópontokon átlagolt mezők
result.nodal_strain           # (csomópontok, 3)
result.nodal_stress           # (csomópontok, 3)
result.nodal_von_mises        # (csomópontok,)
result.nodal_principal_stress # (csomópontok, 2)
result.nodal_principal_angle  # (csomópontok,)
result.nodal_principal_vectors
```

Az alakváltozás- és feszültségértékeket az elemek középpontjában értékeljük ki.
Ezek mellett minden integrációs pont eredménye megmarad. A Quad4 négy
Gauss-ponti értékét az elem csomópontjaira extrapoláljuk, majd a szomszédos
elemek hozzájárulásait átlagoljuk. A Triangle3 állandó alakváltozású, ezért az
egyetlen centroidértéke kerül mindhárom csomópontjára.

## Peremfeltételek és testsúly

```python
# Csak x irányban rögzített csomópont
model.fix_node(0, x=True, y=False)

# Nem nulla előírt elmozdulás
model.prescribe(3, ux=0.1)

# Teljes elnevezett perem: ux=0, uy szabad
model.fix_boundary("left", x=True, y=False)

# Felületi traction és normális nyomás
model.add_boundary_traction("right", tx=2.0, ty=-1.0)
model.add_boundary_pressure("hole", pressure=10.0)

# Gravitáció; ehhez az anyag density értéke nem lehet nulla
model.set_body_acceleration(ay=-9.81)

# A teljes globális konzisztens tömegmátrix külön is lekérhető
mass = model.mass_matrix()
```

Az API nem feltételez mértékegységet: minden adatot egyetlen következetes
mértékegységrendszerben kell megadni.

## Régi modellek használata

```python
from fempy import load_myfem, load_protus

triangle_model = load_myfem("tests/data/myfem/coarse.fem")
triangle_result = triangle_model.solve()

protus_model = load_protus("tests/data/protus/INPUT_FEA_PROTUS.txt")
protus_result = protus_model.solve()
protus_result.write_vtk("protus_uj.vtk")
```

Az importerek megőrzik az anyagot, vastagságot, terheket és megtámasztásokat.
A PROTUS-import a sűrű pszeudoinverz helyett az új ritka megoldót használja.
A VTK-fájl csomóponti és elemközépi mezőket, főfeszültségi vektorokat, valamint
a négy Gauss-pont külön `gp1`–`gp4` cellamezőit is tartalmazza.

## Oktatási térkép

- `fempy/material.py`: Hooke-törvény és a D anyagmátrix;
- `fempy/elements.py`: alakfüggvény-gradiensek, B mátrix és elemmerevség;
- `fempy/mesh.py`: topológia és egyszerű hálógenerálás;
- `fempy/model.py`: globális összeállítás, peremfeltételek és megoldás;
- `fempy/result.py`: típusos eredmény-API;
- `fempy/io.py`, `fempy/plotting.py`: utófeldolgozás;
- `fempy/legacy.py`: a két régi fájlformátum adaptere;
- `examples/`: azonnal futtatható példák;
- `tests/`: anyagmátrix-, patch-, egyensúly- és kompatibilitási tesztek.

## Tesztek

```powershell
python -m pytest
python -m ruff check fempy tests examples
python -m ruff format --check fempy tests examples
```

A patch tesztek ismert lineáris elmozdulásmezőt írnak elő. A végeselemnek ebből
gépi pontossággal kell visszaadnia az állandó alakváltozást. Az egyensúlyteszt
ellenőrzi, hogy a támaszreakciók összege kiegyenlíti a külső terhelést.

## Korlátok

Ez szándékosan kis, átlátható oktatási solver. Jelenleg lineáris statikát,
izotróp anyagot, egy anyagot és egy vastagságot kezel modellenként. Nincs még
kontakt, képlékenység, geometriai nemlinearitás, dinamikai elemzés vagy ipari
minőségű hálógenerátor.
