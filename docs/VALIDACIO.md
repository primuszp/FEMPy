# FEMPy – klasszikus végeselemes validáció

## Mit jelent itt a validáció?

A csomag numerikus implementációját ismert megoldásokkal és elfogadott
benchmarkértékekkel vetjük össze. Pontosabban ez **verifikáció**: azt
ellenőrzi, hogy a diszkretizált egyenleteket és az elemeket helyesen oldjuk-e
meg. Egy valós mérnöki modell validációjához ezen felül mérési eredmény,
anyagazonosítás és az idealizálási hibák vizsgálata is szükséges.

Minden benchmark rögzíti:

- a referenciaértéket;
- a hálósűrűségeket és szabadságfokszámokat;
- a számított eredményt és relatív hibát;
- az elfogadási toleranciát;
- több háló esetén a monoton hibacsökkenést.

## 1. Egytengelyű patch-próba

Egy téglalapot állandó `sigma_x` feszültséggel húzunk. Síkfeszültségben az
egzakt mező:

```text
u_x = sigma_x / E × x
u_y = -nu × sigma_x / E × y
sigma = [sigma_x, 0, 0]
```

A bal perem x irányban rögzített, egyetlen csomópont y irányú rögzítése pedig
csak a merevtest-elmozdulást szünteti meg. A jobb perem terhét az
`add_boundary_traction()` integrálja. Triangle3 és Quad4 esetén is gépi
pontosságú eredményt várunk; a maximális normalizált mezőhiba határa `1e-10`.

Ez egyszerre ellenőrzi az alakfüggvényeket, a B- és D-mátrixot, az
összeállítást, a peremterhelést és az eredmény-visszaállítást.

## 2. Karcsú konzol

A `L=100`, `h=10`, `t=2` konzol végén `P=-1000` eredő megoszló terhelés hat.
Az összehasonlítás a hajlítási és nyírási tagot is tartalmazó
Timoshenko-gerendaeredménnyel történik:

```text
delta = P L³ / (3 E I) + P L / (k G A)
I = t h³ / 12
A = t h
G = E / (2(1 + nu))
k = 5/6
```

A referencia `-9.598095238`. A síkbeli kontinuummodell és a gerendaelmélet
nem teljesen azonos peremközeli állapota miatt itt konvergencia-ellenőrzést és
3,5%-os végső toleranciát használunk. A Quad4 40×8, a lassabban konvergáló
Triangle3 80×16 hálóig fut.

## 3. Cook-membrán

A klasszikus torzításérzékeny feladat négyszögpontjai:

```text
(0, 0), (48, 44), (48, 60), (0, 44)
```

A bal oldal befogott. A 16 hosszú jobb peremen egységnyi eredő függőleges
nyíróterhelés hat. Anyag: `E=1`, `nu=1/3`, vastagság `t=1`, síkfeszültség.
A vizsgált érték a jobb perem közepének, `(48,52)` pontnak a függőleges
elmozdulása. A finomhálós irodalmi referencia `23,9`.

A kompatibilis Quad4 2×2 hálós eredménye `11,85`, a 64×64 értéke körülbelül
`23,92`. Ezeket az értékeket az ENERCALC nyilvános ellenőrzési példája is
közli: <https://media.enercalc.com/ec3d_verification/c-04-%28cook-membrane-problem%29.htm>.

A FEMPy öt hálószintet használ 2×2-től 32×32-ig. Elfogadási határ Quad4
esetén 1%, Triangle3 esetén 3%. Mindkét sorozatnál kötelező, hogy minden
finomítás csökkentse a relatív hibát.

## Futtatás és kimenetek

```powershell
python examples/validate_classic_fem.py
```

A program:

1. lefuttatja mind a hat esetet;
2. PASS/FAIL összefoglalót ír a terminálba;
3. létrehozza a `classic_validation_results.md` részletes táblázatot;
4. elmenti a `classic_validation_convergence.png` logaritmikus hibaábrát;
5. hibakóddal leáll, ha bármely eset túllépi a toleranciát vagy nem konvergál.

Ezért a validáció CI-folyamatban és kiadás előtti regressziós ellenőrzésként
is közvetlenül használható.
