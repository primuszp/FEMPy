# FEMaster2D sample models

These six triangular `.fem` models come from the
[FEMaster2D sample collection](https://github.com/primuszp/FEMaster2D/tree/master/FEMaster/Samples).

They are kept in their original text format and loaded through
`fempy.load_myfem`. They are retained as compatibility and teaching fixtures.

- `coarse.fem`: minimal eight-node teaching model;
- `smallrect.fem`: small irregular rectangular mesh;
- `btfemexample.fem`: larger bracket-like example;
- `rect.fem`: rectangular model;
- `notchedspecimen.fem`: refined notched specimen;
- `plhole.fem`: plate-with-hole model.

Run all six as a coloured gallery with:

```powershell
python examples/colored_femaster_samples.py
```
