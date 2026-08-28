# FEMPy példák

A példák önállóan futtathatók a projekt fejlesztői telepítése után:

```powershell
python -m pip install -e ".[gmsh]"
python examples/cantilever_quad.py
```

Ajánlott tanulási sorrend:

| Példa | Mit tanít? | Gmsh kell? |
|---|---|---:|
| `cantilever_quad.py` | öt lépéses alap-API, megoldás, VTK | nem |
| `colored_cantilever.py` | elmozdulás- és feszültségmezők | nem |
| `boundary_conditions.py` | x/y/kétirányú megtámasztás és terhelésnyíl | igen |
| `gmsh_plate_with_hole.py` | Triangle3 háló, lyuk, helyi finomítás | igen |
| `gmsh_quad_mesh.py` | Quad4 rekombináció és peremcímkék | igen |
| `t6_localized_plots.py` | T6, kvadratikus peremterhelés, magyar/angol tudományos plot | igen |
| `pressurized_hole.py` | normális nyomás a furat peremén | igen |
| `sparse_matrix_visualization.py` | CSR szerkezet és memóriahasználat | nem |
| `validate_classic_fem.py` | patch, konzol és Cook benchmark | nem |
| `protus_compat.py` | régi PROTUS formátum importja | nem |
| `triangle_from_myfem.py` | régi Myfem formátum importja | nem |

A PNG-k a példák aktuális referencia-kimenetei. A `.vtk` eredmények a
`.gitignore` miatt nem kerülnek véletlenül verziókezelésbe.
