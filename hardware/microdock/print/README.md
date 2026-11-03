# MicroDock print plates (Bambu Lab P1S)

The dock doesn't fit one P1S plate (256 × 256 mm): the base and the board are too big to share one. These are three
plates with every part turned for printing. Each plate has a 3MF that opens in Bambu Studio with the parts in place,
an STL of the whole plate for other slicers, and a top view.

| Plate | Parts | Files |
|---|---|---|
| <img src="microdock_plate1_base.png" width="220" alt="Plate 1: the base"> | **1:** base (195 × 200 × 19 mm) | [3MF](microdock_plate1_base.3mf) · [STL](microdock_plate1_base.stl) |
| <img src="microdock_plate2_board_hood_keeper.png" width="220" alt="Plate 2: board, hood and keeper"> | **2:** board on its back, hood upside down, keeper | [3MF](microdock_plate2_board_hood_keeper.3mf) · [STL](microdock_plate2_board_hood_keeper.stl) |
| <img src="microdock_plate3_tag.png" width="220" alt="Plate 3: the two-colour tag"> | **3:** tag, black base with white cells on top | [3MF](microdock_plate3_tag.3mf) · [STL](microdock_plate3_tag.stl) |

## Printing

1. In Bambu Studio, select your P1S, white PLA and a 0.20 mm process, then open a plate's 3MF. Bambu Studio notes
   that the file isn't from Bambu Lab and loads its geometry and colours only: that is expected, and your presets
   stay as they are. The parts land where the top views show them.
2. **Plate 2:** select the hood and turn on supports: type *tree (auto)*, *on build plate only*. They grow under the
   hood's top plate and its pegs. The skids, lips and side walls that the duck slides on face up and stay clean, and
   the ceilings inside the hood bridge (up to about 32 mm).
3. **Plate 3:** the tag's base is black and its cells and border are white. With an AMS the 3MF's two colours map to
   two filaments. Without one, print it in black and add a pause at 1.6 mm to change to white.

About 200 g of PLA in all at normal infill.

Regenerate the plates after changing the design:

```bash
uv run hardware/microdock/source/plates.py
```

These are plain 3MF files without slicer settings; they haven't been sliced or printed yet.
