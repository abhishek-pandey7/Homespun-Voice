"""Generate the site icon: a plain weave, in SVG and PNG.

Homespun is hand-woven cloth, so the mark is a weave: three warp threads, three
weft, interlacing over and under. It survives 16px, which a Devanagari letter
does not - at favicon size the matras close up and any of अ, ह, थ read as the
same grey smudge.

The PNG is rasterised here rather than with an imaging library. A 32x32 icon is
a handful of rectangles, and writing the chunks directly is less weight than a
dependency that exists to draw them.

    python scripts/make_favicon.py
"""

from __future__ import annotations

import pathlib
import struct
import zlib

OUT = pathlib.Path("web")

# Light and dark are both needed: an SVG favicon can carry its own media query,
# and the PNG fallback has to pick one, so it picks the light-background mark
# which reads on either browser chrome.
PAPER = (243, 236, 223)
INK = (29, 92, 72)
INK_DARK = (134, 194, 169)
PAPER_DARK = (22, 18, 14)

SIZE = 32
BAR = 4
POSITIONS = (5, 13, 21)   # three threads each way, evenly spread
RADIUS = 6                # rounded square, matching the page's corner language


def weave_rects() -> list[tuple[int, int, int, int, int]]:
    """Rectangles in paint order, as (x, y, w, h, layer).

    Layer 0 is drawn first. The illusion of interlacing comes from redrawing the
    warp square on top at every other crossing, so a thread passes over one
    neighbour and under the next.
    """
    out: list[tuple[int, int, int, int, int]] = []
    for x in POSITIONS:                       # warp, vertical
        out.append((x, 2, BAR, SIZE - 4, 0))
    for y in POSITIONS:                       # weft, horizontal
        out.append((2, y, SIZE - 4, BAR, 1))
    for i, x in enumerate(POSITIONS):         # warp back on top, alternating
        for j, y in enumerate(POSITIONS):
            if (i + j) % 2 == 0:
                out.append((x, y, BAR, BAR, 2))
    return out


def write_svg() -> pathlib.Path:
    rects = weave_rects()
    body = "\n".join(
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="1" fill="var(--t)"/>'
        for x, y, w, h, _ in rects
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">
  <style>
    :root {{ --b: rgb{PAPER}; --t: rgb{INK}; }}
    @media (prefers-color-scheme: dark) {{
      :root {{ --b: rgb{PAPER_DARK}; --t: rgb{INK_DARK}; }}
    }}
  </style>
  <rect width="{SIZE}" height="{SIZE}" rx="{RADIUS}" fill="var(--b)"/>
{body}
</svg>
'''
    path = OUT / "favicon.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def write_png() -> pathlib.Path:
    """Rasterise the same mark, with a rounded corner mask and no antialiasing.

    At 32px, aliasing on a geometric mark of straight bars is invisible, and
    skipping it keeps this to arithmetic.
    """
    px = [[PAPER for _ in range(SIZE)] for _ in range(SIZE)]

    # Rounded corners: clear anything outside the radius at each corner.
    corners = [(RADIUS, RADIUS), (SIZE - RADIUS - 1, RADIUS),
               (RADIUS, SIZE - RADIUS - 1), (SIZE - RADIUS - 1, SIZE - RADIUS - 1)]
    transparent = set()
    for y in range(SIZE):
        for x in range(SIZE):
            near_x = x < RADIUS or x > SIZE - RADIUS - 1
            near_y = y < RADIUS or y > SIZE - RADIUS - 1
            if near_x and near_y:
                cx, cy = min(corners, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)
                if (cx - x) ** 2 + (cy - y) ** 2 > RADIUS ** 2:
                    transparent.add((x, y))

    for x0, y0, w, h, _ in weave_rects():
        for y in range(y0, min(y0 + h, SIZE)):
            for x in range(x0, min(x0 + w, SIZE)):
                px[y][x] = INK

    raw = bytearray()
    for y in range(SIZE):
        raw.append(0)  # filter type: none
        for x in range(SIZE):
            r, g, b = px[y][x]
            raw += bytes((r, g, b, 0 if (x, y) in transparent else 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    path = OUT / "favicon.png"
    path.write_bytes(png)
    return path


def main() -> int:
    svg, png = write_svg(), write_png()
    print(f"[ok] {svg}  {svg.stat().st_size} bytes")
    print(f"[ok] {png}  {png.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
