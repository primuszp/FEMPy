# FEMPy klasszikus validációs eredmények

Összesített eredmény: **PASS**

| Feladat | Elem | Legfinomabb eredmény | Referencia | Hiba | Tűrés | Eredmény |
|---|---:|---:|---:|---:|---:|---:|
| egytengelyű patch-próba | Quad4 | 6.0025489e-14 | 0 | 6.003e-12% | 1e-08% | PASS |
| karcsú konzol | Quad4 | -9.3311807 | -9.5980952 | 2.781% | 3.5% | PASS |
| Cook-membrán | Quad4 | 23.817634 | 23.9 | 0.3446% | 1% | PASS |
| egytengelyű patch-próba | Triangle3 | 4.0083031e-13 | 0 | 4.008e-11% | 1e-08% | PASS |
| karcsú konzol | Triangle3 | -9.3216606 | -9.5980952 | 2.88% | 3.5% | PASS |
| Cook-membrán | Triangle3 | 23.275122 | 23.9 | 2.615% | 3% | PASS |
| egytengelyű patch-próba | Triangle6 | 7.1433781e-13 | 0 | 7.143e-11% | 1e-08% | PASS |
| karcsú konzol | Triangle6 | -9.5766726 | -9.5980952 | 0.2232% | 3.5% | PASS |
| Cook-membrán | Triangle6 | 23.952198 | 23.96 | 0.03256% | 1% | PASS |

## Konvergenciasorok

### egytengelyű patch-próba – Quad4

Egzakt homogén feszültség- és elmozdulásmező; az eredmény a maximális normalizált mezőhiba.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 8×4 | 90 | 6.00254889e-14 | 6.00255e-12% |

### karcsú konzol – Quad4

A referencia hajlítási és nyírási alakváltozást tartalmazó Timoshenko-csúcselmozdulás.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 10×2 | 66 | -6.77221745 | 29.4421% |
| 20×4 | 210 | -8.66910241 | 9.67893% |
| 40×8 | 738 | -9.33118072 | 2.78091% |

### Cook-membrán – Quad4

Síkfeszültség, E=1, ν=1/3, egységnyi jobb oldali nyíróerő; referencia u_y(48,52)=23,9.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 2×2 | 18 | 11.8451795 | 50.4386% |
| 4×4 | 50 | 18.2991658 | 23.4345% |
| 8×8 | 162 | 22.0791834 | 7.61848% |
| 16×16 | 578 | 23.4304113 | 1.96481% |
| 32×32 | 2178 | 23.817634 | 0.344628% |

### egytengelyű patch-próba – Triangle3

Egzakt homogén feszültség- és elmozdulásmező; az eredmény a maximális normalizált mezőhiba.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 8×4 | 90 | 4.00830314e-13 | 4.0083e-11% |

### karcsú konzol – Triangle3

A referencia hajlítási és nyírási alakváltozást tartalmazó Timoshenko-csúcselmozdulás.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 10×2 | 66 | -3.53417624 | 63.1784% |
| 20×4 | 210 | -6.68115401 | 30.3908% |
| 40×8 | 738 | -8.63336136 | 10.0513% |
| 80×16 | 2754 | -9.32166059 | 2.8801% |

### Cook-membrán – Triangle3

Síkfeszültség, E=1, ν=1/3, egységnyi jobb oldali nyíróerő; referencia u_y(48,52)=23,9.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 2×2 | 18 | 6.74253006 | 71.7886% |
| 4×4 | 50 | 11.2519923 | 52.9205% |
| 8×8 | 162 | 17.3311629 | 27.4847% |
| 16×16 | 578 | 21.5921504 | 9.65627% |
| 32×32 | 2178 | 23.2751219 | 2.61455% |

### egytengelyű patch-próba – Triangle6

Egzakt homogén feszültség- és elmozdulásmező; az eredmény a maximális normalizált mezőhiba.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 8×4 | 306 | 7.14337812e-13 | 7.14338e-11% |

### karcsú konzol – Triangle6

A referencia hajlítási és nyírási alakváltozást tartalmazó Timoshenko-csúcselmozdulás.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 10×2 | 210 | -9.53940112 | 0.611518% |
| 20×4 | 738 | -9.56802269 | 0.313318% |
| 40×8 | 2754 | -9.57667265 | 0.223196% |

### Cook-membrán – Triangle6

Síkfeszültség, E=1, ν=1/3, egységnyi jobb oldali nyíróerő; referencia u_y(48,52)=23,9.

| Háló | Szabadságfok | Eredmény | Relatív hiba |
|---:|---:|---:|---:|
| 2×2 | 50 | 21.2514062 | 11.3046% |
| 4×4 | 162 | 23.4760941 | 2.01964% |
| 8×8 | 578 | 23.8607058 | 0.414416% |
| 16×16 | 2178 | 23.9271249 | 0.137208% |
| 32×32 | 8450 | 23.9521978 | 0.0325636% |
