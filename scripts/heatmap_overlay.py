"""heatmap_overlay.py  —  prints a temp summary every 120 ticks."""

def _report(tick, api):
    if tick % 120 != 0:
        return
    rows, cols = api.rows, api.cols
    total = sum(api.temp(r, c) for r in range(rows) for c in range(cols))
    avg = total / max(1, rows * cols)
    api.print(f"[tick {tick}] avg grid temp: {avg:.1f} °C")

api.on_tick(_report)
