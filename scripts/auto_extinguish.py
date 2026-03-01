"""auto_extinguish.py  —  quenches any cell that reaches stage 3 burn."""

def _quench(event, api):
    r, c = event.get("row", -1), event.get("col", -1)
    if r >= 0:
        api.set_temp(r, c, 20.0)

api.on_event("ignition", _quench)
api.notify("Auto-extinguish active", 3)
