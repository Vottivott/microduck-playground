"""MicroDock v2, the seated dock: the MicroDock's design, fitted to the seated duck.

The duck sits in front of the dock (the official sitstand policy), then butt-walks backward under the hood. Seated, its
trunk rides about 64 mm lower than standing, so everything on the board that meets the hood (its window and V-seats, the
cable notch) moves down with `z0`. Three more changes come from the seated pose:
- the window lets the hood float further sideways (the sit shifts the duck by up to ~25 mm) and drop further (the seated
  height varies more than the standing one);
- the adapter sits 2.3 mm lower in the hood: the duck backs in leaning ~2-3 deg forward and settles at ~7 deg, and the
  lower adapter keeps the magnetic tip on it in both.
So the board and the hood are new; the base, keeper and tag are the MicroDock's.
"""
import os, sys
from dataclasses import dataclass
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "microdock", "source"))
import dock as dk


@dataclass
class Params(dk.Params):
    z0: float = 55.0                # trunk origin height when seated and docked on the plate (measured in simulation)
    hood_float_y: float = 20.0      # the hood's sideways float (the MicroDock: 12)
    hood_drop: float = 12.0         # how far the hood can hang below docked (the MicroDock: 7)
    adapter_dz: float = -2.3        # the adapter this much below the port's height in the hood frame (the MicroDock: 0)
