"""MicroDock v2, the seated dock: the MicroDock's design, fitted to the seated duck.

The duck sits in front of the dock (the official sitstand policy), then scoots backward on its seat under the hood.
Seated, its trunk rides about 57 mm lower than standing, so everything on the board that meets the hood (its window and
V-seats, the cable clip) moves down with `z0`. The window also lets the hood float further sideways (the sit shifts the
duck by up to ~25 mm) and drop further (the seated height varies more than the standing one).
The docking policy sits level, so the adapter stays at the port's height, as on the MicroDock (`adapter_dz` 0; an
earlier policy leaned ~7 deg forward and needed it 2.3 mm lower).
So the board is new; the hood, base, keeper and tag are the MicroDock's.
"""
import os, sys
from dataclasses import dataclass
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "microdock", "source"))
import dock as dk


@dataclass
class Params(dk.Params):
    z0: float = 62.0                # trunk origin height when seated and docked on the plate (measured in simulation)
    hood_float_y: float = 20.0      # the hood's sideways float (the MicroDock: 12)
    hood_drop: float = 12.0         # how far the hood can hang below docked (the MicroDock: 7)
    adapter_dz: float = 0.0         # the adapter this much below the port's height in the hood frame (the MicroDock: 0)
