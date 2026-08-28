# Változásnapló

## 1.1.0 – 2026-08-28

- teljes Triangle6/T6 elem hétpontos Dunavant-integrálással;
- strukturált T6 háló, Triangle3→T6 konverzió és Gmsh `order=2` támogatás;
- görbült T6 peremek hárompontos konzisztens traction- és nyomásintegrálása;
- T6 VTK-export, eredményextrapoláció és négyrészes plot-háromszögelés;
- magyar/angol `PlotStyle`, tizedesvessző/tizedespont és mértékegységek;
- tudományos feszültségjelölések, mérnöki kitevők és előjelhelyes színnormák;
- kilences klasszikus validációs csomag Triangle3, Triangle6 és Quad4 elemekre.

## 1.0.0 – 2026-08-28

- egységes, dokumentált Triangle3 és Quad4 végeselemes API;
- memóriahatékony ritka összeállítás és direkt/CG megoldók;
- modern, opcionális Gmsh Python API sokszögekhez, ívekhez és lyukakhoz;
- név szerinti peremek, irányonkénti megtámasztások, traction és nyomás;
- Matplotlib háló-, peremfeltétel-, eredmény- és ritkamátrix-ábrák;
- VTK export és PROTUS/Myfem kompatibilitási import;
- klasszikus patch-, konzol- és Cook-membrán verifikáció;
- Ruff formázás/lint, GitHub Actions és publikus projekt-dokumentáció.
