"""Tiny SVG plotting utility to avoid third-party dependencies in CI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def _line_plot(rows: List[Dict[str, object]], y_key: str, title: str, out_file: Path):
    width, height = 700, 420
    pad = 60
    budgets = sorted({int(r["budget"]) for r in rows})
    interfaces = sorted({str(r["interface"]) for r in rows})
    yvals = [float(r[y_key]) for r in rows]
    ymin, ymax = min(yvals), max(yvals)
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0

    def xmap(v: int) -> float:
        return pad + (v - budgets[0]) * (width - 2 * pad) / max(1, budgets[-1] - budgets[0])

    def ymap(v: float) -> float:
        return height - pad - (v - ymin) * (height - 2 * pad) / (ymax - ymin)

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="{width//2}" y="24" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>',
    ]

    for i, iface in enumerate(interfaces):
        pts = [r for r in rows if str(r["interface"]) == iface]
        pts = sorted(pts, key=lambda r: int(r["budget"]))
        path = " ".join(f"{xmap(int(r['budget'])):.1f},{ymap(float(r[y_key])):.1f}" for r in pts)
        color = palette[i % len(palette)]
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx, ly = width - 220, pad + 20 + i * 20
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+18}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx+24}" y="{ly+4}" font-size="12">{iface}</text>')

    parts.append("</svg>")
    out_file.write_text("\n".join(parts), encoding="utf-8")


def plot_metrics(rows: List[Dict[str, object]], title_prefix: str, out_prefix: Path):
    _line_plot(rows, "top1_agreement", f"{title_prefix}: Fidelity vs Budget", out_prefix.with_name(out_prefix.name + "_fidelity.svg"))
    _line_plot(rows, "kl_divergence", f"{title_prefix}: KL vs Budget", out_prefix.with_name(out_prefix.name + "_kl.svg"))
