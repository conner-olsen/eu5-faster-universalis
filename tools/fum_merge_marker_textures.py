"""Pre-composite stacked map marker textures into single images.

Each recipe reproduces one vanilla texture stack - base texture plus its
modify_texture tints, overlays and drop shadow - as one image, so the engine
draws one quad with no shader work instead of several. Output is uncompressed
32-bit BGRA with a full mip chain: the map minifies markers heavily, and the
mip chain is what stops them shimmering while the camera pans.

A modify_texture with no blend_mode replaces RGB and keeps the base alpha.
Vanilla's own gold frames prove it: circle_frame.dds and unit_frame.dds are
pure black with an alpha mask, tinted by color_gold_texture with no blend_mode,
and render gold rather than black.

Usage: python tools/fum_merge_marker_textures.py [recipe ...]
"""
import os
import struct
import sys

import numpy as np
from PIL import Image

GAME = r"C:/Steam/steamapps/common/Europa Universalis V/game"
MOD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLATS = os.path.join(MOD, "main_menu", "gfx", "interface", "buttons", "flats")

# The three gfx roots merge into one namespace, so a texture is looked up under
# whichever root holds it.
GFX_ROOTS = ("main_menu", "loading_screen", "in_game")


def load(rel):
    """Load a vanilla gfx path as a float RGBA array in 0..1."""
    for root in GFX_ROOTS:
        path = os.path.join(GAME, root, rel)
        if os.path.exists(path):
            return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
    raise FileNotFoundError(rel)


def resize(img, size):
    a = Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8), "RGBA")
    return np.asarray(a.resize(size, Image.LANCZOS), dtype=np.float64) / 255.0


def colorize(base, rgb):
    """modify_texture with no blend_mode: replace RGB, keep alpha."""
    out = base.copy()
    out[..., :3] = rgb
    return out


def over(dst, src):
    """Straight-alpha source-over composite."""
    sa = src[..., 3:4]
    da = dst[..., 3:4]
    oa = sa + da * (1.0 - sa)
    safe = np.where(oa > 0, oa, 1.0)
    out = np.empty_like(dst)
    out[..., :3] = (src[..., :3] * sa + dst[..., :3] * da * (1.0 - sa)) / safe
    out[..., 3:4] = oa
    return out


def paste(canvas, img, box):
    """Draw img into canvas at (left, top, right, bottom) in canvas texels."""
    left, top, right, bottom = box
    scaled = resize(img, (right - left, bottom - top))
    region = canvas[top:bottom, left:right]
    canvas[top:bottom, left:right] = over(region, scaled)
    return canvas


def blank(w, h):
    return np.zeros((h, w, 4), dtype=np.float64)


def gaussian_alpha(alpha, sigma):
    """Separable Gaussian blur of an alpha plane, used to fake a glow."""
    radius = max(1, int(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    pad = np.pad(alpha, ((0, 0), (radius, radius)), mode="constant")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)
    pad = np.pad(tmp, ((radius, radius), (0, 0)), mode="constant")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


def mip_chain(img):
    """Halve to 1x1. Premultiplied so transparent texels cannot darken edges."""
    levels = [img]
    cur = img
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        h = max(1, cur.shape[0] // 2)
        w = max(1, cur.shape[1] // 2)
        pm = cur.copy()
        pm[..., :3] *= pm[..., 3:4]
        small = resize(pm, (w, h))
        a = small[..., 3:4]
        small[..., :3] = np.where(a > 0, small[..., :3] / np.where(a > 0, a, 1.0), 0.0)
        levels.append(small)
        cur = small
    return levels


def write_dds(path, img):
    """Uncompressed 32-bit BGRA with mips - the format GlorpUI already ships."""
    levels = mip_chain(img)
    h, w = img.shape[:2]
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    # CAPS | HEIGHT | WIDTH | PITCH | PIXELFORMAT | MIPMAPCOUNT
    struct.pack_into("<I", header, 8, 0x1 | 0x2 | 0x4 | 0x8 | 0x1000 | 0x20000)
    struct.pack_into("<I", header, 12, h)
    struct.pack_into("<I", header, 16, w)
    struct.pack_into("<I", header, 20, w * 4)
    struct.pack_into("<I", header, 28, len(levels))
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x1 | 0x40)  # ALPHAPIXELS | RGB
    struct.pack_into("<I", header, 88, 32)
    struct.pack_into("<I", header, 92, 0x00FF0000)
    struct.pack_into("<I", header, 96, 0x0000FF00)
    struct.pack_into("<I", header, 100, 0x000000FF)
    struct.pack_into("<I", header, 104, 0xFF000000)
    struct.pack_into("<I", header, 108, 0x1000 | 0x400000 | 0x8)
    body = bytearray()
    for lv in levels:
        rgba = (np.clip(lv, 0, 1) * 255).round().astype(np.uint8)
        body += rgba[..., [2, 1, 0, 3]].tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(bytes(header) + bytes(body))
    print("%-42s %dx%d  %d mips  %d bytes" % (os.path.basename(path), w, h, len(levels), 128 + len(body)))


# --------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------

def construction_marker_background():
    """Re-export the hand-authored marker background with a mip chain.

    The pixels are kept exactly as authored; only the container changes. The
    shipped file is DXT1, whose 1-bit alpha cut the circle edge to a hard
    stair-step, and it carries no mips at all.
    """
    src = os.path.join(FLATS, "fum_construction_marker_background.dds")
    img = np.asarray(Image.open(src).convert("RGBA"), dtype=np.float64) / 255.0
    write_dds(src, img)


def port_marker():
    """Flatten port_marker's two icons and its drop shadow into one image.

    Vanilla draws sort_maritime_presence at 100% and alpha 0.5 with a
    real-time 2.5px black glow, then maritime_presence at 102%. The widget is
    30x30, so the canvas is 36x36 logical to hold the glow, drawn at 4x.

    The engine's glow kernel is not documented, so the shadow is approximated
    with a Gaussian sized to reach the canvas edge and no further. This is the
    one recipe here that is not exact - compare it against vanilla in game.
    """
    scale = 4
    pad = 3  # logical px each side, and the extent the shadow is fitted to
    box = 30
    canvas_px = (box + pad * 2) * scale
    inner = box * scale
    off = pad * scale

    under = load("gfx/interface/icons/flat_icons/geopolitics/sort_maritime_presence.dds")
    over_icon = load("gfx/interface/icons/flat_icons/geopolitics/maritime_presence.dds")

    canvas = blank(canvas_px, canvas_px)

    # Drop shadow: blurred alpha of the lower icon, black. Sigma is a third of
    # the padding so three sigma lands on the canvas edge, and the shadow
    # carries the lower icon's own 0.5 alpha rather than full strength.
    shadow_src = resize(under, (inner, inner))
    plane = np.zeros((canvas_px, canvas_px))
    plane[off:off + inner, off:off + inner] = shadow_src[..., 3]
    glow = blank(canvas_px, canvas_px)
    glow[..., 3] = np.clip(gaussian_alpha(plane, pad * scale / 3.0), 0, 1) * 0.5
    canvas = over(canvas, glow)

    lower = under.copy()
    lower[..., 3] *= 0.5
    canvas = paste(canvas, lower, (off, off, off + inner, off + inner))

    top = int(round(box * 1.02 * scale))
    o2 = (canvas_px - top) // 2
    canvas = paste(canvas, over_icon, (o2, o2, o2 + top, o2 + top))

    write_dds(os.path.join(FLATS, "fum_port_marker.dds"), canvas)


def paper_card_bg_blue():
    """Bake the city label card's colour tint into the card texture.

    simple_paper_card.dds is pure black with an alpha mask, tinted by
    color_bg_blue_texture. The tint is a uniform colour with no condition, so
    baking it is exact at every label width and the 9-slice geometry is
    unchanged.
    """
    card = load("gfx/interface/cards/simple_paper_card.dds")
    tint = load("gfx/interface/colors/bg_blue.dds")[0, 0, :3]
    write_dds(os.path.join(FLATS, "fum_paper_card_bg_blue.dds"), colorize(card, tint))


RECIPES = {
    "construction_marker_background": construction_marker_background,
    "port_marker": port_marker,
    "paper_card_bg_blue": paper_card_bg_blue,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(RECIPES)
    for name in wanted:
        RECIPES[name]()
