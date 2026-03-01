"""lava_rain.py  —  rains lava from the top row every 30 ticks."""
import random as _r

def _rain(tick, api):
    if tick % 30 == 0:
        c = _r.randint(0, api.cols - 1)
        api.set(0, c, api.material_id("lava"))

api.on_tick(_rain)
api.notify("Lava rain active!", 3)
