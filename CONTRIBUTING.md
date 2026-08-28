# Hozzájárulás a FEMPy projekthez

Köszönjük, ha hibajavítással, dokumentációval vagy új verifikációs példával
segíted a projektet.

## Fejlesztői környezet

```powershell
git clone https://github.com/primuszp/FEMPy.git
cd FEMPy
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,gmsh]"
```

## Beküldés előtt

```powershell
python -m ruff format --check fempy tests examples
python -m ruff check fempy tests examples
python -m pytest
python examples/validate_classic_fem.py
```

Új numerikus funkcióhoz kérünk legalább egy kis, kézzel vagy analitikusan
ellenőrizhető tesztet. Új elemformulához patch-próba és hálókonvergencia is
szükséges. A publikus API kapjon magyar docstringet és rövid példát.

Ne keverj egy pull requestbe egymástól független formázási és numerikus
változtatásokat. A hibajelentés tartalmazza a Python-verziót, a minimális
reprodukáló kódot, a várt és a tényleges eredményt.
