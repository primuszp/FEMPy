# FEMPy Edu – részletes kódmagyarázat

Ez a dokumentum azt mutatja be, hogyan jut el a program a geometriától a
színezett feszültségábráig. A forráskód minden publikus eleme magyar docstringet
tartalmaz, ezért interaktívan is lekérdezhető:

```python
from fempy import Model, Quad4, SolverOptions

help(Model)
help(Quad4)
help(SolverOptions)
```

## 1. A publikus API gondolati modellje

A felhasználói kód öt, egymásra épülő fogalmat követ:

```text
Mesh + Material
       │
       ▼
     Model  ← terhek és peremfeltételek
       │
       ▼
  ritka Kff rendszer
       │
       ▼
AnalysisResult
       │
       ├── numerikus tömbök
       ├── VTK export
       └── Matplotlib ábrák
```

Minimalista példa:

```python
from fempy import LinearElasticMaterial, Model, rectangular_quad_mesh

mesh = rectangular_quad_mesh(20, 5, width=200.0, height=50.0)
steel = LinearElasticMaterial(young_modulus=210_000.0, poisson_ratio=0.3)
model = Model(mesh, steel, thickness=5.0, name="konzol")

model.fix_nodes(mesh.nodes_where(x=0.0))
right = mesh.nodes_where(x=200.0)
model.add_nodal_loads(right, fy=-1_000.0 / len(right))

result = model.solve()
```

## 2. `material.py` – az anyagtörvény

A `LinearElasticMaterial` tárolja a Young-modulust, Poisson-tényezőt és
sűrűséget. A `constitutive_matrix()` állítja elő a Hooke-törvény `D` mátrixát:

```text
sigma = D epsilon

epsilon = [epsilon_x, epsilon_y, gamma_xy]
sigma   = [sigma_x, sigma_y, tau_xy]
```

Síkfeszültségnél `sigma_z = 0`; ez vékony lemez tipikus modellje.
Síkalakváltozásnál `epsilon_z = 0`; ez hosszú test keresztmetszetére alkalmas.

A `von_mises()` nem egyszerűen az x–y komponenseket kezeli: plane strain
esetén visszaállítja a `sigma_z` feszültséget is.

## 3. `elements.py` – lokális végeselemes matematika

### `Triangle3`

A háromcsomópontos CST elem alakfüggvénye lineáris, ezért a gradiense és a `B`
mátrixa az egész elemen állandó. Az elemi merevség:

```text
Ke = thickness × area × Bᵀ D B
```

Egyetlen centroid-integrációs pont pontos. A centroid feszültsége mindhárom
elemcsomópontra változatlanul kerül.

### `Quad4`

A négycsomópontos elem bilineáris alakfüggvényeket használ a `[-1,1]²`
naturális tartományban. A kód minden Gauss-pontban:

1. kiszámítja a naturális alakfüggvény-gradienst;
2. felépíti a Jacobi-mátrixot;
3. ellenőrzi, hogy a Jacobi-determináns pozitív;
4. a gradienst fizikai x–y koordinátákba transzformálja;
5. összeállítja a `B` mátrixot;
6. hozzáadja a `Bᵀ D B det(J)` hozzájárulást.

A négy pont koordinátája `±1/sqrt(3)`. Az eredmények csomópontokra történő
extrapolációja a Gauss-ponti alakfüggvénymátrix inverzével történik.

### Konzisztens tömegmátrix

Mindkét elem megvalósítja a `mass_matrix()` metódust. Quad4 esetén:

```text
Me = integral(rho × thickness × Nᵀ N × det(J))
```

A testsúly csomóponti erővektora `Me @ acceleration`.

## 4. `mesh.py` – geometria és topológia

A `Mesh` csak a következőket tárolja:

- `(node_count, 2)` koordinátatömb;
- Triangle3/Quad4 elemek tuple-je;
- opcionális, név szerinti csomópont- és peremélhalmazok.

A konstruktor ellenőrzi:

- a tömb alakját és végességét;
- minden elem csomópontindexét;
- a pozitív elempozíciót és területet.

A koordinátatömb írásvédetté válik. A `nodes_where()` koordinátavonal alapján
keres peremcsomópontokat, ezért nem kell kézzel csomópontszámokat felsorolni.

### `geometry.py` – hálózótól független geometria

A `Geometry2D` nem tartalmaz Gmsh-specifikus azonosítókat. Pontokat, egyenes
szakaszokat, köríveket és zárt hurkokat tárol. Pontosan egy külső hurok és
tetszőleges számú belső lyuk lehet benne. A görbékhez rendelt emberi nevek
jelentik a kapcsolatot a geometria és a későbbi peremfeltételek között.

```python
geometry = (
    Geometry2D("plate")
    .add_rectangle(100.0, 50.0)
    .add_circle((50.0, 25.0), 8.0, boundary="hole")
    .set_boundary_size("hole", 1.0)
)
```

Az `add_polygon()` általános sokszöget fogad. A pontokat nem kell záró
kezdőponttal megismételni. A `boundary_names` lista minden sokszögoldalhoz egy
nevet rendel. Tetszőleges íves kontúrhoz az `add_loop()` folytonos
`LineSegment2D` és `CircularArc2D` szegmenseket fogad. A közös végpontokat egy
közös topológiai pontként tárolja, ezért a létrejövő Gmsh-hurok folytonos.

### `gmsh.py` – közvetlen Gmsh Python API

A `GmshMesher.generate()` lépései:

1. ellenőrzi a `Geometry2D` topológiáját;
2. Gmsh pontokat, egyeneseket, köríveket és síkfelületet hoz létre;
3. a peremnevekből egy dimenziós fizikai csoportokat készít;
4. alkalmazza a globális és helyi elemméreteket;
5. memóriában legenerálja a kétdimenziós hálót;
6. a Gmsh csomópontcímkéit tömör, nullától induló FEMPy indexekké alakítja;
7. a fizikai peremeket `node_sets` és `edge_sets` alakban visszaadja.

Csak a kétdimenziós elemek által ténylegesen használt csomópontok kerülnek a
`Mesh` objektumba. Ez kizárja a körközépponthoz hasonló geometriai segédpontok
szabad szabadságfokait, amelyek különben szingulárissá tennék a rendszert.

A Gmsh importja késleltetett. Emiatt a csomag minden más része telepített Gmsh
nélkül is működik; hálózási kéréskor pedig a hibaüzenet megadja a szükséges
`pip install -e ".[gmsh]"` parancsot.

### Név szerinti peremfeltételek

`Model.fix_boundary("left")` a hálóban tárolt név alapján rögzíti az összes
érintett csomópontot. Az `add_boundary_traction()` az állandó felületi terhet
konzisztens csomóponti erővé integrálja. Egy `L` hosszú kétcsomópontos él két
végpontjára egyenként

```text
f_node = traction × thickness × L / 2
```

jut. A szomszédos élek közös csomóponti hozzájárulásai összeadódnak.

Az `add_boundary_pressure()` minden peremélhez megkeresi az egyetlen
szomszédos elemet. Az él középpontja és az elem centroidja alapján választja
ki a kifelé mutató normálist, majd a pozitív nyomást ennek ellentétes, befelé
mutató irányában integrálja. Ez a geometriai vizsgálat furatnál is helyes, és
nem függ a peremél Gmsh által visszaadott irányától.

## 5. `model.py` – globális összeállítás

### Szabadságfok-sorrend

Minden csomópont két szabadságfoka egymás mellett szerepel:

```text
[u1, v1, u2, v2, u3, v3, ...]
```

A `prescribe()` a kötött globális szabadságfokhoz előírt értéket tárol. A
`fix_node()` ennek nulla értékű kényelmi változata.

### Miért nem készül teljes `K` a `solve()` közben?

A hagyományos út:

```text
teljes K összeállítás → K[free][:, free] másolat → megoldás
```

Ez egyidejűleg tárolhatja a teljes és a redukált mátrixot. A FEMPy közvetlenül
a redukált rendszert állítja össze:

```text
Kff uf = ff - Kfc uc
```

Az `_assemble_reduced_system()` két menetet végez:

1. topológia alapján pontosan megszámolja a szükséges szabad–szabad elemi
   bejegyzéseket;
2. előre lefoglalt NumPy-tömbökbe írja a COO sor-, oszlop- és értékadatokat.

A kötött oszlopok nem kerülnek a mátrixba. Hatásuk elemenként a jobb oldalból
vonódik le. A COO→CSR átalakítás összeadja a közös globális pozícióra kerülő
elemi hozzájárulásokat.

### Reakcióerők teljes `K` nélkül

Megoldás után a program elemenként számítja:

```text
internal_element_force = Ke @ ue
reaction = assembled_internal_force - external_force
```

Így a reakciókhoz sem szükséges a teljes globális merevségi mátrix.

### Mikor használjuk a `stiffness_matrix()` metódust?

Oktatáshoz, hibakereséshez és vizualizációhoz:

```python
K = model.stiffness_matrix()
Kff = model.stiffness_matrix(reduced=True)
```

Mindkettő SciPy CSR-mátrix, soha nem NumPy sűrű tömb.

## 6. `solver.py` – ritka egyenletmegoldás

### Automatikus mód

```python
result = model.solve()
```

Az `auto` mód 20 000 szabad szabadságfokig ritka direkt megoldót választ.
Nagyobb rendszernél Jacobi-előkondicionált konjugált gradiensre vált.

### Direkt solver

A ritka direkt megoldás robusztus és pontos, de a faktorizáció során fill-in
keletkezhet. A faktorok memóriaigénye nagyobb lehet az eredeti CSR-mátrixnál.

### Konjugált gradiens

A CG szimmetrikus pozitív definit mátrixra készült. Nem faktorizál, csak
mátrix-vektor szorzásokat végez. A Jacobi-előkondicionáló:

```text
M⁻¹ = diag(Kff)⁻¹
```

`LinearOperator` formában működik, ezért még külön diagonális ritka mátrixot
sem tárol.

```python
from fempy import SolverOptions

result = model.solve(
    SolverOptions(
        method="cg",
        relative_tolerance=1e-8,
        max_iterations=50_000,
        preconditioner="jacobi",
    )
)
```

A `result.solver_info` tartalmazza az iterációszámot, maradékot, nemnulla
elemszámot és a CSR tömbök tényleges memóriaigényét.

## 7. `result.py` – utófeldolgozás

Minden elemhez létrejön:

- elemközépi alakváltozás és feszültség;
- minden Gauss-pont teljes eredménye;
- elemcsomópontokra extrapolált alakváltozás és feszültség.

A hálószintű `nodal_stress` minden globális csomópontnál átlagolja a hozzá
kapcsolódó elemek extrapolált értékeit. Ebből számítható a csomóponti von Mises,
főfeszültség és főirány.

Az `AnalysisResult` tömbjei:

```python
result.displacement             # node_count × 2
result.reaction                 # node_count × 2
result.stress                   # element_count × 3
result.nodal_stress             # node_count × 3
result.principal_stress         # element_count × 2
result.nodal_principal_stress   # node_count × 2
```

## 8. `plotting.py` – ábrák és ritka mátrixok

A `result.plot()` elem- és csomóponti mezőket is kezel. A Quad4 elemeket csak a
megjelenítéshez két háromszögre bontja; ez a numerikus modellt nem módosítja.

A merevségi mátrix mintája:

```python
model.plot_stiffness_matrix(kind="sparsity")
model.plot_stiffness_matrix(kind="magnitude", reduced=True)
```

A `plot_sparse_matrix()` a COO sor- és oszlopindexeket közvetlenül rajzolja.
Nincs `toarray()`, ezért egy 100 000 × 100 000 méretű ritka mátrixból sem készít
óriási kétdimenziós NumPy-tömböt. A `max_points` a Matplotlib pontszámát korlátozza.

## 9. `legacy.py` és `io.py` – kompatibilitás

`load_protus()` Quad4 modellt, `load_myfem()` Triangle3 modellt készít a régi
szövegfájlokból. A számozás, anyag, terhelés és megtámasztás automatikusan
átalakul az új API szerkezetévé.

A `write_vtk()` a következőket exportálja:

- elmozdulás és reakció;
- csomóponti és elemfeszültségek;
- alakváltozások és von Mises-értékek;
- főfeszültségek és irányvektorok;
- külön Gauss-ponti mezők.

## 10. Ajánlott kódolvasási sorrend

1. `examples/cantilever_quad.py`
2. `fempy/mesh.py`
3. `fempy/material.py`
4. `fempy/elements.py`
5. `fempy/model.py`
6. `fempy/solver.py`
7. `fempy/result.py`
8. `fempy/plotting.py`
9. `examples/sparse_matrix_visualization.py`

## 11. Új elemtípus hozzáadása

Egy új elemnek az `Element2D` protokoll metódusait kell megvalósítania:

- `area()`;
- `stiffness()`;
- `strain_at_center()`;
- `integration_data()`;
- `extrapolate_to_nodes()`;
- `mass_matrix()`.

Ezután a `Mesh` és a `Model` változtatás nélkül képes használni, amennyiben az
elem két szabadságfokot rendel minden csomóponthoz.
