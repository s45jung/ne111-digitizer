"""NE111 Plot Digitizer

A Streamlit app that pulls x-y data out of a picture of a graph.

How to use it:
1. Upload an image of a plot (png, jpg, gif, tiff, bmp or webp, colour or greyscale).
2. Calibrate: click two points you know on the x-axis (X1, X2) and two on the
   y-axis (Y1, Y2), then type in their values. Log axes work too, and because
   the calibration is a full affine fit, rotated or skewed scans are fine.
3. Digitize: click on the data points. Each dataset gets its own name and colour.
4. Download the CSV (row 1 = dataset names, row 2 = axis labels, then the values).
"""

import copy
import csv
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage
from streamlit_image_coordinates import streamlit_image_coordinates


# ---- Constants ----

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a5c8",
           "#f032e6", "#9a6324", "#469990", "#800000", "#808000", "#000075"]
CAL_KEYS = ["X1", "X2", "Y1", "Y2"]
CAL_COLORS = {"X1": "#d62728", "X2": "#d62728", "Y1": "#1f77b4", "Y2": "#1f77b4"}
CAL_HELP = {
    "X1": "a point on the **x-axis** whose value you know (e.g. the first labelled tick)",
    "X2": "a second **x-axis** point, as far from X1 as possible",
    "Y1": "a point on the **y-axis** whose value you know (e.g. the lowest labelled tick)",
    "Y2": "a second **y-axis** point, as far from Y1 as possible",
}
MODE_CAL, MODE_ADD, MODE_TRACE, MODE_DEL = "Calibrate", "Add points", "Auto-trace", "Delete points"
MODES = [MODE_CAL, MODE_ADD, MODE_TRACE, MODE_DEL]
IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp", "webp"]
LINEAR, LOG, RECIPROCAL, SQRT, SQUARED, DATE = (
    "Linear", "Logarithmic", "Reciprocal (1/x)", "Square root", "Squared", "Date / time")
AXIS_SCALES = [LINEAR, LOG, RECIPROCAL, SQRT, SQUARED, DATE]

# Starting values for the widgets. Settings about one graph's axes go back to
# these when a new image is loaded; preferences are kept.
GRAPH_DEFAULTS = {"xlabel": "x", "ylabel": "y", "xunit": "", "yunit": "",
                  "xscale": LINEAR, "yscale": LINEAR,
                  "X1_val": "0", "X2_val": "10", "Y1_val": "0", "Y2_val": "10",
                  "x_step": "", "y_step": "", "zoom": 1.0, "pan_x": 0.5, "pan_y": 0.5,
                  "sort_x": False}
PREFERENCES = {"straighten": True, "snap": True, "show_grid": True, "connect": False,
               "round_prec": True, "trace_n": 25, "trace_tol": 60, "adv_tools": False}
WIDGET_DEFAULTS = {**GRAPH_DEFAULTS, **PREFERENCES}
RENDER_W = 1200  # width the plot is drawn at before the browser scales it down
MAX_HISTORY = 100


# ---- Loading images ----

def to_rgb(im):
    """Convert any PIL image (palette, 16-bit, alpha, CMYK, animated...) to 8-bit RGB.

    Returns the RGB image and a dict of info about the original file.
    """
    info = {"format": im.format or "image", "mode": im.mode,
            "frames": getattr(im, "n_frames", 1)}
    if info["frames"] > 1:
        im.seek(0)  # first frame of an animated GIF / multi-page TIFF
    try:
        im = ImageOps.exif_transpose(im)  # phone photos store their rotation in EXIF
    except Exception:
        pass

    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        # 16-bit or float image: stretch the values to fill 0-255
        arr = np.asarray(im, dtype=np.float64)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        im = Image.fromarray(np.nan_to_num(arr).astype(np.uint8), "L")
    if im.mode == "1":
        im = im.convert("L")
    if im.mode in ("P", "PA"):
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        # put transparent images on a white background
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg

    rgb = im.convert("RGB")
    info["greyscale"] = is_greyscale(rgb)
    info["size"] = rgb.size
    return rgb, info


def is_greyscale(rgb):
    """True if R, G and B are (almost) equal everywhere, checked on a small copy."""
    small = np.asarray(rgb.resize((min(200, rgb.width), min(200, rgb.height))), dtype=np.int16)
    # differences between neighbouring channels: G-R and B-G
    return bool(np.max(np.abs(np.diff(small, axis=2))) < 10)


def decode_image(data):
    """Open image bytes and return (rgb_image, info)."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return to_rgb(im)


def make_sample_plot():
    """PNG bytes of a semi-log plot with two curves, scanned slightly crooked (3°)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=110)
    ax.semilogy(x, 2 * np.exp(0.5 * x), "o-", color="black", label="Reactor A")
    ax.semilogy(x, 50 * np.exp(0.2 * x), "s--", color="dimgray", label="Reactor B")
    ax.set_xlim(0, 10)
    ax.set_ylim(1, 1e4)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Activity (Bq)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    im = im.rotate(3, expand=True, fillcolor="white", resample=Image.BICUBIC)
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


# ---- Axis scales ----

def to_axis(values, scale, name="Value"):
    """Data values -> evenly spaced positions along the axis.

    On a log axis equal steps in log10(x) are equal distances, on a reciprocal
    axis it's 1/x, and so on. Dates are stored as seconds, which are already even.
    """
    v = np.asarray(values, dtype=float)
    if scale == LOG:
        if np.any(v <= 0):
            raise ValueError(f"{name} must be greater than 0 on a logarithmic axis.")
        return np.log10(v)
    if scale == RECIPROCAL:
        if np.any(v == 0):
            raise ValueError(f"{name} can't be 0 on a reciprocal axis.")
        return 1 / v
    if scale == SQRT:
        if np.any(v < 0):
            raise ValueError(f"{name} can't be negative on a square-root axis.")
        return np.sqrt(v)
    if scale == SQUARED:
        if np.any(v < 0):
            raise ValueError(f"{name} can't be negative on a squared axis.")
        return v ** 2
    return v  # linear, and dates (in seconds)


def from_axis(positions, scale):
    """Positions along the axis -> data values (the reverse of to_axis).

    Positions where the scale has no value (e.g. below 0 on a square-root
    axis) come back as nan.
    """
    u = np.asarray(positions, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if scale == LOG:
            return 10.0 ** u
        if scale == RECIPROCAL:
            return 1 / u
        if scale == SQRT:
            return np.where(u >= 0, u ** 2, np.nan)
        if scale == SQUARED:
            return np.sqrt(u)
    return u


def parse_date(text):
    """A typed date/time -> seconds. Accepts 2024-03-01, 2024-03-01 14:30, 01/03/2024..."""
    text = str(text).strip()
    if not any(c in text for c in "-/:"):
        raise ValueError(f"'{text}' isn't a date. Type it like 2024-03-01 or 2024-03-01 14:30.")
    try:
        return pd.Timestamp(text).timestamp()
    except ValueError:
        raise ValueError(f"'{text}' isn't a date I can read. Type it like 2024-03-01 14:30.")


def parse_value(text, scale):
    """A typed axis value: a date on a date axis, otherwise a number."""
    return parse_date(text) if scale == DATE else parse_number(text)


def fmt_value(v, scale=LINEAR):
    """A value as text: dates as 2024-03-01 (plus the time if it isn't midnight)."""
    if scale != DATE:
        return fmt(v)
    if not math.isfinite(v):
        return ""
    t = pd.Timestamp(v, unit="s")
    if t.hour == t.minute == t.second == 0:
        return t.strftime("%Y-%m-%d")
    return t.strftime("%Y-%m-%d %H:%M" if t.second == 0 else "%Y-%m-%d %H:%M:%S")


# ---- Calibration ----

def fit_axis(p1, p2, along_other, v1, v2):
    """Solve for (a, b, c) so that value = a*col + b*row + c.

    The value must be v1 at p1 and v2 at p2, and must not change when you
    move parallel to the other axis.
    """
    M = [[p1[0], p1[1], 1],
         [p2[0], p2[1], 1],
         [along_other[0], along_other[1], 0]]
    return np.linalg.solve(M, [v1, v2, 0])


class Calibration:
    """Converts between image pixels and data values.

    The fit is [X', Y'] = A @ [col, row] + off, where X' and Y' are positions
    along each axis (see to_axis: e.g. log10 x on a log axis). It's a general
    affine map, so rotation, skew and different scales on each axis are all
    handled, and the four points don't need to share an origin.
    """

    def __init__(self, px, vals, xscale=LINEAR, yscale=LINEAR):
        """px: pixel position of X1, X2, Y1, Y2. vals: their values (dates in seconds)."""
        missing = [k for k in CAL_KEYS if px.get(k) is None]
        if missing:
            raise ValueError("Place calibration point(s): " + ", ".join(missing) + ".")
        X1, X2, Y1, Y2 = [np.asarray(px[k], dtype=float) for k in CAL_KEYS]
        scale_of = {"X1": xscale, "X2": xscale, "Y1": yscale, "Y2": yscale}
        v = {k: float(to_axis(vals[k], scale_of[k], k)) for k in CAL_KEYS}

        if v["X1"] == v["X2"]:
            raise ValueError("X1 and X2 must have different values.")
        if v["Y1"] == v["Y2"]:
            raise ValueError("Y1 and Y2 must have different values.")
        if np.allclose(X1, X2):
            raise ValueError("X1 and X2 are at the same spot. Place them further apart.")
        if np.allclose(Y1, Y2):
            raise ValueError("Y1 and Y2 are at the same spot. Place them further apart.")

        try:
            a, b, c = fit_axis(X1, X2, Y2 - Y1, v["X1"], v["X2"])
            d, e, f = fit_axis(Y1, Y2, X2 - X1, v["Y1"], v["Y2"])
        except np.linalg.LinAlgError:
            raise ValueError("The x-axis and y-axis points are parallel. "
                             "Check that X1/X2 are on the x-axis and Y1/Y2 on the y-axis.")
        A = np.array([[a, b], [d, e]])
        if abs(np.linalg.det(A)) < 1e-15:
            raise ValueError("Calibration is degenerate. Re-place the points.")

        self.A = A
        self.off = np.array([c, f])
        self.scales = [xscale, yscale]
        self.centre = np.mean([X1, X2, Y1, Y2], axis=0)  # middle of the plot, roughly

    def to_linear(self, pts):
        """Pixel positions -> positions along each axis (the fitted space)."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        return pts @ self.A.T + self.off

    def from_linear(self, vals):
        """Positions along each axis -> pixel positions."""
        vals = np.asarray(vals, dtype=float).reshape(-1, 2)
        return (vals - self.off) @ np.linalg.inv(self.A).T

    def to_data(self, pts):
        """Pixel positions (N x 2) -> data values (N x 2)."""
        vals = self.to_linear(pts)
        for col, scale in enumerate(self.scales):
            vals[:, col] = from_axis(vals[:, col], scale)
        return vals

    def to_pixel(self, vals):
        """Data values (N x 2) -> pixel positions (N x 2)."""
        v = np.asarray(vals, dtype=float).reshape(-1, 2).copy()
        for col, (scale, name) in enumerate(zip(self.scales, ["x values", "y values"])):
            v[:, col] = to_axis(v[:, col], scale, name)
        return self.from_linear(v)


def line_tilt(p, q, vertical):
    """Angle (degrees, counter-clockwise positive) of the line p->q from horizontal
    or vertical. Returns None if either point is missing."""
    if p is None or q is None:
        return None
    dx, dy = q[0] - p[0], q[1] - p[1]
    # image rows grow downward, hence the minus signs
    if vertical:
        ang = math.degrees(math.atan2(-dx, -dy))
    else:
        ang = math.degrees(math.atan2(-dy, dx))
    return (ang + 90) % 180 - 90  # fold into [-90, 90)


def axis_tilt(px):
    """Tilt of the x-axis from horizontal and of the y-axis from vertical."""
    return (line_tilt(px.get("X1"), px.get("X2"), False),
            line_tilt(px.get("Y1"), px.get("Y2"), True))


# ---- Typing in numbers ----

SUPERSCRIPTS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
                "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}


def parse_number(text):
    """Read a typed value such as 250, -2.5, 1e4, 10^4, 10**-3, 2x10^3, 10⁴ or 1,000.

    Raises ValueError if it can't be read.
    """
    s = str(text).strip().lower().replace(" ", "").replace(",", "").replace("−", "-")
    # turn superscripts into powers: 10⁻³ -> 10^-3
    out = ""
    in_power = False
    for ch in s:
        if ch in SUPERSCRIPTS:
            if not in_power:
                out += "^"
                in_power = True
            out += SUPERSCRIPTS[ch]
        else:
            out += ch
            in_power = False
    s = out.replace("**", "^").replace("×", "*").replace("x", "*")
    if s == "":
        raise ValueError("nothing typed in")

    try:
        total = 1.0
        for factor in s.split("*"):  # 2*10^3 -> "2", "10^3"
            top, _, bottom = factor.partition("/")  # allow simple fractions like 1/3
            if "^" in top:
                base, power = top.split("^")
                value = float(base) ** float(power)
            else:
                value = float(top)
            if bottom:
                value = value / float(bottom)
            total = total * value
    except (ValueError, ZeroDivisionError, OverflowError):
        raise ValueError(f"'{text}' isn't a number I can read. Try 250, 1e4 or 10^4.")
    if not math.isfinite(total):
        raise ValueError(f"'{text}' is too big.")
    return total


# ---- Straightening ----

def background_color(img):
    """Most images have a plain border; use its median colour as the background."""
    a = np.asarray(img, dtype=float)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(c) for c in np.median(border, axis=0))


def detect_tilt(img, max_angle=10.0):
    """Estimate how far the plot is rotated, in degrees (counter-clockwise positive).

    Axes and gridlines are long straight lines of dark pixels. If we count the
    dark pixels in each row (and column) after rotating their coordinates, the
    counts are spikiest when we've rotated by exactly the plot's tilt.
    """
    small = img.convert("L")
    small.thumbnail((600, 600))
    rows, cols = np.nonzero(np.asarray(small) < 128)
    if len(rows) < 50:
        return 0.0
    if len(rows) > 40000:  # plenty for a good estimate, and much faster
        pick = np.random.default_rng(0).choice(len(rows), 40000, replace=False)
        rows, cols = rows[pick], cols[pick]

    def spikiness(deg):
        t = math.radians(deg)
        score = 0.0
        for proj in [cols * math.sin(t) + rows * math.cos(t),   # along horizontal lines
                     cols * math.cos(t) - rows * math.sin(t)]:  # along vertical lines
            counts = np.bincount(np.round(proj - proj.min()).astype(int))
            score += np.sum(counts.astype(float) ** 2)
        return score

    best = 0.0
    for step, span in [(0.5, max_angle), (0.05, 0.5)]:  # coarse search, then fine
        angles = np.arange(best - span, best + span + step / 2, step)
        scores = [spikiness(a) for a in angles]
        best = float(angles[int(np.argmax(scores))])
    return round(best, 2)


def rotate_image(img, angle):
    """Rotate counter-clockwise by angle degrees, growing the canvas to fit."""
    if angle == 0:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=True,
                      fillcolor=background_color(img))


def rotate_points(pts, angle, old_size, new_size):
    """Where points land when an old_size image is rotated by angle into new_size."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    t = math.radians(angle)
    d = pts - np.array(old_size) / 2  # relative to the old centre
    # counter-clockwise on screen, remembering image rows grow downward
    x = d[:, 0] * math.cos(t) + d[:, 1] * math.sin(t)
    y = -d[:, 0] * math.sin(t) + d[:, 1] * math.cos(t)
    return np.column_stack([x, y]) + np.array(new_size) / 2


# ---- Snapping clicks onto markers ----

def ink_map(img):
    """How different each pixel is from the background (0 = background)."""
    a = np.asarray(img, dtype=np.float32)
    bg = np.array(background_color(img), dtype=np.float32)
    return np.sqrt(np.sum((a - bg) ** 2, axis=2))


def snap_to_marker(ink, px, py, radius, threshold=180):
    """Move a click to the centre of the marker or line it's on, if there is one.

    threshold skips faint things like light gridlines. A filled marker is
    thicker than any line touching it, so its centre is the pixel deepest
    inside the ink (found with a distance transform); this isn't pulled
    sideways by a line or axis running out of it. Thin lines and hollow
    markers have no thick part, so for those we use the centre of mass.
    """
    h, w = ink.shape
    x0, x1 = max(0, int(px - radius)), min(w, int(px + radius) + 1)
    y0, y1 = max(0, int(py - radius)), min(h, int(py + radius) + 1)
    if x1 <= x0 or y1 <= y0:
        return px, py
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5  # pixel centres
    inside = (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
    strong = ink[y0:y1, x0:x1] > threshold

    depth = ndimage.distance_transform_edt(np.pad(strong, 1))[1:-1, 1:-1]
    depth[~inside] = 0
    if depth.max() >= 2.5:  # a filled marker
        core = depth >= depth.max() - 0.5
        return float(xx[core].mean()), float(yy[core].mean())

    weight = np.where(inside, np.clip(ink[y0:y1, x0:x1] - threshold, 0, None), 0)
    if weight.sum() == 0:
        return px, py
    return float((weight * xx).sum() / weight.sum()), float((weight * yy).sum() / weight.sum())


# ---- Calibration check grid ----

def nice_ticks(lo, hi, n=8):
    """About n round-number tick values between lo and hi (steps of 1, 2 or 5 x 10^k)."""
    if hi < lo:
        lo, hi = hi, lo
    raw = (hi - lo) / n
    if raw <= 0:
        return []
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in [1, 2, 5, 10]:
        if m * mag >= raw:
            step = m * mag
            break
    first = math.ceil(lo / step) * step
    return [round(v, 12) for v in np.arange(first, hi + step / 2, step) if v <= hi]


def date_ticks(lo, hi):
    """Round times (every minute, hour, day, week...) between lo and hi seconds."""
    steps = [60, 300, 900, 3600, 3 * 3600, 6 * 3600, 86400, 2 * 86400, 7 * 86400,
             30 * 86400, 365 * 86400]
    step = steps[-1]
    for s in steps:
        if s >= (hi - lo) / 8:
            step = s
            break
    return list(np.arange(math.ceil(lo / step) * step, hi + 1, step, dtype=float))


def axis_ticks(lo, hi, scale):
    """Gridline positions along an axis that runs from lo to hi (in fitted space)."""
    lo, hi = min(lo, hi), max(lo, hi)
    if scale == LINEAR:
        return nice_ticks(lo, hi)
    if scale == DATE:
        return date_ticks(lo, hi)
    if scale == LOG and hi - lo >= 1:  # whole decades: 1, 10, 100...
        return [float(k) for k in range(math.ceil(lo), math.floor(hi) + 1)]
    # other scales: round data values, then placed where they belong on the axis
    ends = from_axis([lo, hi], scale)
    if not np.all(np.isfinite(ends)):
        return []
    ticks = []
    for v in nice_ticks(min(ends), max(ends)):
        try:
            u = float(to_axis(v, scale))
        except ValueError:  # e.g. 0 on a reciprocal axis
            continue
        if lo <= u <= hi:
            ticks.append(u)
    return ticks


def calibration_grid(cal, size, calib):
    """Gridlines from the calibration, as (start_px, end_px, label, label_px, axis).

    If the calibration is right, these sit exactly on the plot's own gridlines.
    """
    w, h = size
    corners = cal.to_linear([(0, 0), (w, 0), (0, h), (w, h)])
    xlo, xhi = corners[:, 0].min(), corners[:, 0].max()
    ylo, yhi = corners[:, 1].min(), corners[:, 1].max()
    # labels go along the axes the user clicked
    y_axis_line = cal.to_linear([calib["X1"]])[0, 1]
    x_axis_line = cal.to_linear([calib["Y1"]])[0, 0]

    lines = []
    for axis, (lo, hi), scale in [("x", (xlo, xhi), cal.scales[0]), ("y", (ylo, yhi), cal.scales[1])]:
        for v in axis_ticks(lo, hi, scale):
            text = fmt_value(float(from_axis(v, scale)), scale)
            if axis == "x":
                ends = cal.from_linear([(v, ylo), (v, yhi)])
                spot = cal.from_linear([(v, y_axis_line)])[0]
            else:
                ends = cal.from_linear([(xlo, v), (xhi, v)])
                spot = cal.from_linear([(x_axis_line, v)])[0]
            lines.append((tuple(ends[0]), tuple(ends[1]), text, tuple(spot), axis))
    return lines


# ---- Finding the axes automatically ----

def tick_centres(band, skip, lo, hi):
    """Tick marks along one side of an axis.

    band: dark-pixel counts per column (or row) in a 4-pixel strip beside the
    axis. A tick fills (almost) the whole strip. skip: frame lines to ignore.
    """
    strong = band >= 3
    for c in skip:
        strong[max(0, c - 2): c + 3] = False
    strong[:max(0, lo)] = False
    strong[hi:] = False
    labels, n = ndimage.label(strong)
    if n == 0:
        return []
    centres = ndimage.center_of_mass(strong, labels, range(1, n + 1))
    return [c[0] + 0.5 for c in centres]


def regular_ticks(cands, tol=2.5):
    """Longest run of evenly spaced positions among the candidates.

    Real tick marks are evenly spaced (decades too, on a log axis); stray hits
    from a curve or a label usually aren't, so they get left out.
    """
    cands = sorted(set(round(c, 1) for c in cands))
    best = cands[:1]
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            step = b - a
            if step < 8:  # too close together to be separate ticks
                continue
            run = [a, b]
            for c in cands:
                if c > run[-1] and abs(c - (run[-1] + step)) <= tol:
                    run.append(c)
            longer = len(run) > len(best)
            wider = len(run) == len(best) and run[-1] - run[0] > best[-1] - best[0]
            if longer or wider:
                best = run
    return best


def find_ticks(img):
    """Find the axes and their tick marks on a (straightened) plot.

    The x-axis is the lowest long horizontal line and the y-axis the left-most
    long vertical line. Returns (x-axis row, y-axis column, x tick columns,
    y tick rows, right end of the x-axis, top end of the y-axis).
    """
    dark = np.asarray(img.convert("L")) < 160
    rows, cols = dark.sum(axis=1), dark.sum(axis=0)
    if rows.max() < 0.3 * dark.shape[1] or cols.max() < 0.3 * dark.shape[0]:
        raise ValueError("Couldn't find the axis lines. Place the points by hand.")
    long_rows = np.nonzero(rows >= 0.6 * rows.max())[0]
    long_cols = np.nonzero(cols >= 0.6 * cols.max())[0]
    xr, yc = int(long_rows.max()), int(long_cols.min())
    x_end = int(np.nonzero(dark[xr])[0].max()) + 1
    y_top = int(np.nonzero(dark[:, yc])[0].min())

    # ticks may point out of the plot or into it: try a strip on each side
    x_ticks, y_ticks = [], []
    for off, skip in [(2, []), (-6, long_cols)]:
        found = tick_centres(dark[xr + off: xr + off + 4, :].sum(axis=0), skip, yc - 2, x_end + 2)
        if len(found) > len(x_ticks):
            x_ticks = found
    for off, skip in [(2, []), (-6, long_rows)]:
        found = tick_centres(dark[:, yc - off - 3: yc - off + 1].sum(axis=1), skip, y_top - 2, xr + 3)
        if len(found) > len(y_ticks):
            y_ticks = found
    # where the axes cross is often a tick too (e.g. the origin)
    x_ticks = regular_ticks(x_ticks + [yc + 0.5])
    y_ticks = regular_ticks(y_ticks + [xr + 0.5])
    return xr, yc, x_ticks, y_ticks, x_end, y_top


def detect_axes(img):
    """Suggest X1, X2 on the first and last x-axis ticks, Y1, Y2 on the lowest and highest y ticks."""
    xr, yc, x_ticks, y_ticks, x_end, y_top = find_ticks(img)
    if len(x_ticks) < 2:  # no ticks: use the ends of the axis line instead
        x_ticks = [yc + 0.5, x_end - 0.5]
    if len(y_ticks) < 2:
        y_ticks = [y_top + 0.5, xr + 0.5]
    return {"X1": (x_ticks[0], xr + 0.5), "X2": (x_ticks[-1], xr + 0.5),
            "Y1": (yc + 0.5, y_ticks[-1]), "Y2": (yc + 0.5, y_ticks[0])}


def tick_points(img):
    """Pixel positions of every tick mark on both axes (empty if none are found)."""
    try:
        xr, yc, x_ticks, y_ticks, _, _ = find_ticks(img)
    except ValueError:
        return []
    return [(t, xr + 0.5) for t in x_ticks] + [(yc + 0.5, t) for t in y_ticks]


def nearest_tick(ticks, px, py, radius):
    """The tick mark closest to (px, py) if it's within radius, else the click itself."""
    best, best_d = (px, py), radius
    for tx, ty in ticks:
        d = math.hypot(tx - px, ty - py)
        if d <= best_d:
            best, best_d = (tx, ty), d
    return best


# ---- Cleaning up the numbers ----

def round_to_step(v, step):
    """Round v to the largest power of ten that's no bigger than step.

    step is how much one pixel is worth, so e.g. 0.0124132 with a step of
    0.015 becomes 0.01: the digits after that are just mouse wobble.
    """
    if not step > 0 or not math.isfinite(step):
        return v
    p = 10 ** math.floor(math.log10(step))
    return float(f"{round(v / p) * p:.12g}")  # .12g drops float noise like 40.300000000000004


def pixel_step(cal, col, v):
    """How much the value v changes over one pixel along its axis."""
    res = np.hypot(*cal.A[col])  # fitted units per pixel
    scale = cal.scales[col]
    try:
        u = float(to_axis(v, scale))
    except ValueError:
        return float("nan")
    return abs(float(from_axis(u + res, scale)) - v)


def clean_values(data, cal, round_prec, x_step, y_step):
    """Tidy digitized values: snap to multiples of x_step / y_step (if given),
    otherwise round to the precision one pixel of the image allows."""
    data = np.array(data, dtype=float).reshape(-1, 2)
    steps = [x_step, y_step]
    for col in range(2):
        for i, v in enumerate(data[:, col]):
            if not math.isfinite(v):
                continue
            if steps[col] and cal.scales[col] != DATE:
                data[i, col] = float(f"{round(v / steps[col]) * steps[col]:.12g}")
            elif round_prec:
                step = pixel_step(cal, col, v)
                if cal.scales[col] == DATE:  # round to a whole second/minute/hour/day
                    unit = 1
                    for u in [60, 3600, 86400]:
                        if u <= step:
                            unit = u
                    data[i, col] = round(v / unit) * unit
                else:
                    data[i, col] = round_to_step(v, step)
    return data


def precision_text(cal):
    """How much one pixel is worth on each axis, e.g. "x ±0.015 · y ±2.0%"."""
    middle = cal.to_data([cal.centre])[0]
    parts = []
    for col, name in enumerate("xy"):
        scale = cal.scales[col]
        step = pixel_step(cal, col, middle[col])
        if not math.isfinite(step):
            continue
        if scale == LOG:
            parts.append(f"{name} ±{100 * step / middle[col]:.1f}%")
        elif scale == DATE:
            parts.append(f"{name} ±{step / 3600:.2g} h" if step >= 600 else f"{name} ±{step:.2g} s")
        elif scale == LINEAR:
            parts.append(f"{name} ±{step:.2g}")
        else:
            parts.append(f"{name} ±{step:.2g} mid-plot")
    return " · ".join(parts)


# ---- Tracing a curve automatically ----

def long_runs(mask, length):
    """True where a pixel is part of a horizontal run of at least `length` on-pixels.

    Uses running totals (cumsum), so it's fast even for very long runs:
    a window of `length` pixels is all on when its total equals `length`.
    """
    h, w = mask.shape
    if length < 1 or length > w:
        return np.zeros_like(mask)
    total = np.zeros((h, w + 1), dtype=np.int32)
    total[:, 1:] = np.cumsum(mask, axis=1)
    full = total[:, length:] - total[:, :-length] == length  # window starting at each column
    # pixel j is covered if a full window starts anywhere from j-length+1 to j
    starts = np.zeros((h, full.shape[1] + 1), dtype=np.int32)
    starts[:, 1:] = np.cumsum(full, axis=1)
    j = np.arange(w)
    hi = np.minimum(j, w - length) + 1
    lo = np.maximum(j - length + 1, 0)
    return starts[:, hi] - starts[:, lo] > 0


def remove_long_lines(mask, frac=0.25):
    """Blank out long horizontal/vertical runs (axes, frame) and the ticks on them."""
    h, w = mask.shape
    lines = long_runs(mask, int(frac * w)) | long_runs(mask.T, int(frac * h)).T
    # also clear 6 pixels either side, where the tick marks are
    return mask & ~ndimage.maximum_filter(lines, size=13)


def follow_curve(lin, seed, step_x, step_y):
    """Walk along the curve one pixel-wide slice at a time, starting at the click.

    lin: the curve's pixels in fitted (x, y) space. In each slice they may form
    several separate blobs (a marker, another curve crossing, a stray label);
    we take the blob closest to where the curve is heading, so the walk stays
    on the curve that was clicked.
    """
    bins = np.floor((lin[:, 0] - seed[0]) / step_x).astype(int)
    order = np.argsort(bins)
    bins, ys = bins[order], lin[order, 1]
    edges = np.nonzero(np.diff(bins))[0] + 1
    slices = {}
    first_of_each = bins[np.concatenate([[0], edges])]
    for b, y in zip(first_of_each, np.split(ys, edges)):
        y = np.sort(y)
        gaps = np.nonzero(np.diff(y) > 3 * step_y)[0] + 1  # split into separate blobs
        slices[int(b)] = np.array([blob.mean() for blob in np.split(y, gaps)])

    path = {}

    def recent_slope(direction):
        """Slope (y per slice) of a straight-line fit to the last 20 slices walked.

        A fit over a stretch of curve isn't thrown off when the walk passes
        through a marker, which is flatter and taller than the line itself.
        """
        keys = sorted(path)
        keys = keys[-20:] if direction == 1 else keys[:20]
        if len(keys) < 3:
            return 0.0
        return np.polyfit(keys, [path[k] for k in keys], 1)[0]

    def walk(b, last_b, last, direction):
        while abs(b - last_b) < 15:  # 15 empty slices in a row: the curve has ended
            if b in slices:
                gap = abs(b - last_b)
                slope = recent_slope(direction)
                guess = last + slope * (b - last_b)
                blobs = slices[b]
                best = blobs[np.argmin(np.abs(blobs - guess))]
                # the allowed distance from the guess grows a little over gaps
                if abs(best - guess) < (8 + 0.5 * gap) * step_y + abs(slope) * gap:
                    path[b] = best
                    last_b, last = b, best
            b += direction

    walk(0, 0, seed[1], 1)  # to the right of the click...
    if not path:
        return np.empty((0, 2))
    # ...then to the left, starting where the right-hand walk began
    first = min(path)
    walk(first - 1, first, path[first], -1)

    keys = sorted(path)
    return np.array([[seed[0] + (k + 0.5) * step_x, path[k]] for k in keys])


def trace_curve(img, ink, click, cal, calib, n_points, tol=60):
    """Trace the curve under a click. Returns (curve colour, n_points pixel positions).

    1. Take the colour of the inkiest pixel near the click.
    2. Keep pixels of about that colour, on the plot side of both axes,
       without the axis lines, joined into one piece (small gaps like dashes bridged).
    3. Follow that piece left and right, then pick n_points evenly spaced in x.
    """
    a = np.asarray(img, dtype=np.float32)
    r = 10  # forgiving: the click only has to be near the curve
    x0, y0 = max(0, int(click[0]) - r), max(0, int(click[1]) - r)
    patch = ink[y0: int(click[1]) + r + 1, x0: int(click[0]) + r + 1]
    if patch.size == 0 or patch.max() < 60:
        raise ValueError("No curve under that click. Click right on a line.")
    iy, ix = np.unravel_index(np.argmax(patch), patch.shape)
    py, px = y0 + iy, x0 + ix
    colour = a[py, px]

    mask = ((a - colour) ** 2).sum(axis=2) < tol ** 2  # pixels of about that colour
    mask = remove_long_lines(mask)

    # only keep the plot side of each axis (drops tick labels, axis titles...)
    rows, cols = np.nonzero(mask)
    lin = cal.to_linear(np.column_stack([cols + 0.5, rows + 0.5]))
    axis_x = cal.to_linear([calib["Y1"]])[0, 0]  # the y-axis is at this x
    axis_y = cal.to_linear([calib["X1"]])[0, 1]  # the x-axis is at this y
    seed = cal.to_linear([(px + 0.5, py + 0.5)])[0]
    keep = ((np.sign(lin[:, 0] - axis_x) == np.sign(seed[0] - axis_x))
            & (np.sign(lin[:, 1] - axis_y) == np.sign(seed[1] - axis_y)))
    mask[rows[~keep], cols[~keep]] = False

    # join nearby pixels (bridges dashes and markers), then keep the clicked piece
    labels, _ = ndimage.label(ndimage.maximum_filter(mask, size=9))
    if labels[py, px] == 0:
        raise ValueError("Couldn't pick out that curve. Try clicking another part of it.")
    lin = lin[keep & (labels[rows, cols] == labels[py, px])]

    path = follow_curve(lin, seed, np.hypot(*cal.A[0]), np.hypot(*cal.A[1]))
    if len(path) < 2:
        raise ValueError("That curve is too short to trace.")
    xs = np.linspace(path[0, 0], path[-1, 0], n_points)
    ys = np.interp(xs, path[:, 0], path[:, 1])
    return tuple(int(c) for c in colour), cal.from_linear(np.column_stack([xs, ys]))


def axis_title(label, unit):
    """"Time" and "h" -> "Time (h)"; no unit -> just the label."""
    label, unit = label.strip(), unit.strip()
    return f"{label} ({unit})" if unit else label


# ---- Re-plotting the result ----

def make_replot(export, xlabel, ylabel, xscale, yscale):
    """Matplotlib figure of the digitized data on the same kind of axes as the original."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def column(values, scale):
        values = np.asarray(values, dtype=float)
        return pd.to_datetime(values, unit="s") if scale == DATE else values

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor("#1c1c1e")
    ax.set_facecolor("#000000")
    for ds in export:
        if len(ds["data"]) == 0:
            continue
        data = np.asarray(ds["data"], dtype=float)
        ax.plot(column(data[:, 0], xscale), column(data[:, 1], yscale), "o-", color=ds["color"],
                label=ds["name"], markersize=5, linewidth=1.2,
                markeredgecolor="white", markeredgewidth=0.8)
    for scale, set_scale in [(xscale, ax.set_xscale), (yscale, ax.set_yscale)]:
        if scale == LOG:
            set_scale("log")
        elif scale in (RECIPROCAL, SQRT, SQUARED):
            def forward(v, scale=scale):
                with np.errstate(all="ignore"):
                    return to_axis_unchecked(v, scale)

            def inverse(u, scale=scale):
                return from_axis(u, scale)
            set_scale("function", functions=(forward, inverse))
    if xscale == DATE:
        fig.autofmt_xdate()
    ax.set_xlabel(xlabel, color="#c7c7cc")
    ax.set_ylabel(ylabel, color="#c7c7cc")
    ax.tick_params(colors="#8e8e93")
    for spine in ax.spines.values():
        spine.set_color("#3a3a3c")
    ax.grid(True, which="both", color="#2c2c2e", linewidth=0.6)
    ax.legend(facecolor="#1c1c1e", edgecolor="#3a3a3c", labelcolor="white", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def to_axis_unchecked(values, scale):
    """Like to_axis, but returns nan for values the scale can't show instead of an error
    (matplotlib asks about values outside the data range too)."""
    v = np.asarray(values, dtype=float)
    if scale == RECIPROCAL:
        return np.where(v != 0, 1 / np.where(v != 0, v, 1), np.nan)
    if scale == SQRT:
        return np.where(v >= 0, np.sqrt(np.abs(v)), np.nan)
    if scale == SQUARED:
        return np.where(v >= 0, v ** 2, np.nan)
    return v


# ---- CSV export ----

def fmt(v):
    return f"{v:.6g}"


def csv_rows(datasets, xlabel, ylabel, xscale=LINEAR, yscale=LINEAR):
    """Rows of the CSV as lists of strings (empty datasets are left out).

    Row 1: dataset names. Row 2: x/y labels. Then x, y pairs side by side,
    padded with blanks where a dataset runs out of points.
    """
    filled = [ds for ds in datasets if len(ds["data"]) > 0]
    names, labels = [], []
    for ds in filled:
        names += [ds["name"], ""]
        labels += [xlabel, ylabel]
    rows = [names, labels]

    n_rows = max([len(ds["data"]) for ds in filled], default=0)
    for i in range(n_rows):
        row = []
        for ds in filled:
            if i < len(ds["data"]):
                x, y = ds["data"][i]
                row += [fmt_value(x, xscale), fmt_value(y, yscale)]
            else:
                row += ["", ""]
        rows.append(row)
    return rows


def rows_to_text(rows):
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


# ---- Drawing ----

def load_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow can't resize the default font
        return ImageFont.load_default()


LABEL_FONT = load_font(18)
GRID_FONT = load_font(15)


def compute_view(W, H, zoom, pan_x, pan_y):
    """Visible part (x0, y0, w, h) of a W x H image, in original pixels."""
    cw, ch = W / zoom, H / zoom
    x0 = min(max(pan_x * W - cw / 2, 0.0), W - cw)
    y0 = min(max(pan_y * H - ch / 2, 0.0), H - ch)
    return x0, y0, cw, ch


def render_view(img, view, disp_w, calib, datasets, active, highlight=None, connect=False,
                grid=None):
    """Crop img to view, scale it to disp_w wide and draw the markers on top."""
    x0, y0, cw, ch = view
    s = disp_w / cw
    disp_h = max(1, round(ch * s))
    resample = Image.LANCZOS if s < 1 else Image.BICUBIC
    base = img.resize((disp_w, disp_h), resample, box=(x0, y0, x0 + cw, y0 + ch)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    def to_screen(p):
        return (p[0] - x0) * s, (p[1] - y0) * s

    # check grid from the calibration: should sit on the plot's own gridlines
    for start, end, text, spot, axis in grid or []:
        d.line([to_screen(start), to_screen(end)], fill="#00a2ff80", width=1)
        sx, sy = to_screen(spot)
        # labels just above the x-axis / just right of the y-axis
        pos = (sx + 4, sy - 6) if axis == "x" else (sx + 6, sy - 4)
        d.text(pos, text, fill="#0077cc", font=GRID_FONT, anchor="lb",
               stroke_width=3, stroke_fill="white")

    # calibration axes, then the four markers
    for k1, k2 in [("X1", "X2"), ("Y1", "Y2")]:
        if calib.get(k1) and calib.get(k2):
            d.line([to_screen(calib[k1]), to_screen(calib[k2])], fill=CAL_COLORS[k1] + "99", width=2)
    for k in CAL_KEYS:
        if calib.get(k) is None:
            continue
        cx, cy = to_screen(calib[k])
        col = CAL_COLORS[k]
        r = 14 if k == highlight else 10
        # a circle with a cross through it
        d.line([(cx - r, cy), (cx + r, cy)], fill=col, width=2)
        d.line([(cx, cy - r), (cx, cy + r)], fill=col, width=2)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)
        d.text((cx + r + 3, cy - r - 12), k, fill=col, font=LABEL_FONT,
               stroke_width=3, stroke_fill="white")

    # data points; the active dataset is drawn last (on top) and bigger
    order = [i for i in range(len(datasets)) if i != active]
    if active < len(datasets):
        order.append(active)
    for i in order:
        ds = datasets[i]
        pts = [to_screen(p) for p in ds["points"]]
        if connect and len(pts) > 1:
            d.line(pts, fill=ds["color"] + "aa", width=2)
        r = 7 if i == active else 5.5
        for cx, cy in pts:
            d.ellipse([cx - r - 1.5, cy - r - 1.5, cx + r + 1.5, cy + r + 1.5], fill="#000000aa")
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ds["color"], outline="white", width=2)
    return Image.alpha_composite(base, layer).convert("RGB")


def nearest_point(datasets, px, py, tol):
    """(dataset index, point index) of the closest point within tol, or None."""
    best, best_d = None, tol
    for i, ds in enumerate(datasets):
        for j, (qx, qy) in enumerate(ds["points"]):
            dist = math.hypot(qx - px, qy - py)
            if dist <= best_d:
                best, best_d = (i, j), dist
    return best


# ---- Session state ----

ss = st.session_state


def new_dataset():
    """A new empty dataset with the first colour that isn't already used."""
    ss.ds_counter = ss.get("ds_counter", 0) + 1
    n = ss.ds_counter
    used = {ds["color"] for ds in ss.get("datasets", [])}
    color = PALETTE[(n - 1) % len(PALETTE)]  # if every colour is taken
    for c in PALETTE:
        if c not in used:
            color = c
            break
    ss.uid_counter = ss.get("uid_counter", 0) + 1  # never reset, so ids are never reused
    return {"uid": f"d{ss.uid_counter}", "name": f"Dataset {n}", "color": color, "points": []}


def clear_calibration():
    ss.calib = {k: None for k in CAL_KEYS}
    ss.calib_next = "X1"
    ss.mode = MODE_CAL


def reset_work():
    """Start over (used when a new image is loaded)."""
    clear_calibration()
    ss.ds_counter = 0
    ss.datasets = []  # empty first so new_dataset() sees no used colours
    ss.datasets = [new_dataset()]
    ss.active = 0
    ss.history = []
    ss.last_click = None
    ss.editor_ver = 0
    for k, v in GRAPH_DEFAULTS.items():
        ss[k] = v
    ss.rotation = 0.0  # degrees the working image is rotated from the upload


def init_state():
    for k, v in WIDGET_DEFAULTS.items():
        if k in ss:
            ss[k] = ss[k]  # keeps the value alive while its widget is hidden
        else:
            ss[k] = v
    ss.setdefault("img_hash", None)
    ss.setdefault("use_sample", False)
    if "datasets" not in ss:
        reset_work()


def snapshot():
    """Save the current work so it can be undone."""
    ss.history.append(copy.deepcopy(
        {"calib": ss.calib, "datasets": ss.datasets, "active": ss.active,
         "ds_counter": ss.ds_counter}))
    del ss.history[:-MAX_HISTORY]


def undo():
    if ss.history:
        snap = ss.history.pop()
        for k, v in snap.items():
            ss[k] = v
        ss.editor_ver += 1


def scale_of(k):
    """The scale chosen for the axis that calibration point k ("X1", "Y2"...) is on."""
    return ss.xscale if k[0] == "X" else ss.yscale


def current_calibration():
    """(Calibration, None) if the axes are calibrated, else (None, error message)."""
    vals = {}
    for k in CAL_KEYS:
        try:
            vals[k] = parse_value(ss[f"{k}_val"], scale_of(k))
        except ValueError as e:
            return None, f"{k} value: {e}"
    try:
        return Calibration(ss.calib, vals, ss.xscale, ss.yscale), None
    except ValueError as e:
        return None, str(e)


def change_rotation(data, new_angle):
    """Straightening was switched on or off: move every placed point to match."""
    original = cached_decode(data)[0]
    old_size = cached_rotate(data, ss.rotation).size
    new_size = cached_rotate(data, new_angle).size

    def move(p):
        p = rotate_points([p], -ss.rotation, old_size, original.size)  # undo the old rotation
        p = rotate_points(p, new_angle, original.size, new_size)        # apply the new one
        return [float(v) for v in p[0]]

    for k in CAL_KEYS:
        if ss.calib[k] is not None:
            ss.calib[k] = tuple(move(ss.calib[k]))
    for ds in ss.datasets:
        ds["points"] = [move(p) for p in ds["points"]]
    ss.rotation = new_angle
    ss.history = []  # old snapshots are in the old orientation
    ss.editor_ver += 1


# ---- Callbacks ----

def cb_copy(src, dst):
    ss[dst] = ss[src]


def cb_set(**kw):
    for k, v in kw.items():
        ss[k] = v


def cb_add_dataset():
    snapshot()
    ss.datasets.append(new_dataset())
    ss.active = len(ss.datasets) - 1
    ss.mode = MODE_ADD


def cb_delete_dataset(i):
    snapshot()
    ss.datasets.pop(i)
    if not ss.datasets:
        ss.datasets.append(new_dataset())
    ss.active = min(ss.active, len(ss.datasets) - 1)
    ss.editor_ver += 1


def cb_remove_points(i, only_last):
    """Remove the last point of dataset i, or all of them."""
    if ss.datasets[i]["points"]:
        snapshot()
        if only_last:
            ss.datasets[i]["points"].pop()
        else:
            ss.datasets[i]["points"] = []
        ss.editor_ver += 1


def cb_auto_axes(data):
    try:
        found = detect_axes(cached_rotate(data, ss.rotation))
    except ValueError as e:
        st.toast(str(e), icon="⚠️")
        return
    snapshot()
    ss.calib = found
    ss.calib_next = None
    st.toast("Axes found. Now type the values of the ticks marked X1, X2, Y1 and Y2.")


def cb_move_dataset(i, step):
    """Move dataset i up (step=-1) or down (+1) in the list and in the CSV."""
    j = i + step
    if 0 <= j < len(ss.datasets):
        snapshot()
        ss.datasets[i], ss.datasets[j] = ss.datasets[j], ss.datasets[i]
        ss.active = j


def cb_sort_dataset(i):
    """Put dataset i's points in order of increasing x."""
    cal, _ = current_calibration()
    pts = ss.datasets[i]["points"]
    if cal is None or len(pts) < 2:
        return
    snapshot()
    order = np.argsort(cal.to_data(pts)[:, 0], kind="stable")
    ss.datasets[i]["points"] = [pts[k] for k in order]
    ss.editor_ver += 1


def cb_reverse_dataset(i):
    snapshot()
    ss.datasets[i]["points"].reverse()
    ss.editor_ver += 1


def cb_merge_dataset(i, other_uid):
    """Add the points of another dataset onto the end of dataset i, then remove it."""
    for j, ds in enumerate(ss.datasets):
        if ds["uid"] == other_uid and j != i:
            snapshot()
            ss.datasets[i]["points"] += ds["points"]
            ss.datasets.pop(j)
            ss.active = i if i < j else i - 1
            ss.editor_ver += 1
            return


def cb_reset_calibration():
    snapshot()
    clear_calibration()


def handle_click(px, py, tol):
    """Do whatever the current mode does with a click at pixel (px, py)."""
    if ss.mode == MODE_CAL:
        target = ss.calib_next
        if target is None:
            st.toast("All four calibration points are placed. Pick one on the right to move it.")
            return
        snapshot()
        ss.calib[target] = (px, py)
        # next point to place: the first empty one after this, wrapping round
        i = CAL_KEYS.index(target)
        ss.calib_next = None
        for k in CAL_KEYS[i + 1:] + CAL_KEYS[:i]:
            if ss.calib[k] is None:
                ss.calib_next = k
                break
    elif ss.mode == MODE_ADD:
        cal, err = current_calibration()
        if cal is None:
            st.toast(f"Finish calibration first: {err}", icon="⚠️")
            return
        snapshot()
        ss.datasets[ss.active]["points"].append([px, py])
        ss.editor_ver += 1
    elif ss.mode == MODE_DEL:
        hit = nearest_point(ss.datasets, px, py, tol)
        if hit is None:
            st.toast("No point close to where you clicked.")
            return
        snapshot()
        i, j = hit
        ss.datasets[i]["points"].pop(j)
        ss.active = i
        ss.editor_ver += 1


# ---- Small display helpers ----

# coloured-circle emoji, so a dataset's colour can be shown in plain text
EMOJI_COLOURS = {"🔴": (220, 40, 50), "🟠": (245, 130, 40), "🟡": (250, 210, 40),
                 "🟢": (60, 180, 75), "🔵": (50, 110, 220), "🟣": (145, 50, 180),
                 "🟤": (150, 100, 40), "⚫": (20, 20, 20), "⚪": (240, 240, 240)}


def colour_dot(hex_colour):
    """The coloured-circle emoji closest to a "#rrggbb" colour."""
    rgb = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    best, best_d = "⚪", None
    for emoji, ref in EMOJI_COLOURS.items():
        d = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if best_d is None or d < best_d:
            best, best_d = emoji, d
    return best


# ---- Sidebar ----

@st.cache_data(show_spinner=False)
def cached_sample():
    return make_sample_plot()


# Images and arrays are never changed after they're made, so cache_resource can
# hand out the same object every time instead of a fresh copy (cache_data copies).
@st.cache_resource(show_spinner=False, max_entries=4)
def cached_decode(data):
    return decode_image(data)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_tilt(data):
    return detect_tilt(cached_decode(data)[0])


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_rotate(data, angle):
    return rotate_image(cached_decode(data)[0], angle)


@st.cache_data(show_spinner=False, max_entries=20)
def cached_replot(export, xlabel, ylabel, xscale, yscale):
    # matplotlib is the slowest part of a rerun, so only redraw when the data changes
    return make_replot(export, xlabel, ylabel, xscale, yscale)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_ticks(data, angle):
    return tick_points(cached_rotate(data, angle))


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_ink(data, angle):
    return ink_map(cached_rotate(data, angle))


def sidebar_image():
    """File uploader / sample buttons. Returns (image bytes or None, file name stem)."""
    st.sidebar.header("Image")
    up = st.sidebar.file_uploader(
        "Upload a plot image", type=IMAGE_TYPES, label_visibility="collapsed",
        help="PNG, JPG, GIF, TIFF, BMP or WebP. Colour or greyscale.")
    if up is not None:
        return up.getvalue(), Path(up.name).stem

    c1, c2 = st.sidebar.columns(2)
    c1.button("Sample", on_click=cb_set, kwargs={"use_sample": True}, width="stretch",
              help="A semi-log plot scanned 3° crooked")
    if ss.use_sample:
        c2.button("Close", on_click=cb_set, kwargs={"use_sample": False}, width="stretch")
        return cached_sample(), "sample_plot"
    return None, ""


def times_text(z):
    return f"{z:g}×"


def sidebar_settings(img, info, tilt):
    kind = "greyscale" if info["greyscale"] else "colour"
    frames = f" · frame 1 of {info['frames']}" if info["frames"] > 1 else ""
    w, h = info["size"]
    st.sidebar.caption(f"{info['format']} · {w}×{h} px · {kind}{frames}")
    if abs(tilt) >= 0.2:
        st.sidebar.toggle(f"Auto-straighten · {abs(tilt):.1f}° tilt found", key="straighten",
                          help="Rotates the image so the axes are level. Points you've "
                               "already placed move with it.")
    else:
        st.sidebar.caption("Image is already straight.")

    st.sidebar.header("Axis labels")
    c1, c2 = st.sidebar.columns([2, 1])
    c1.text_input("x-axis label", key="xlabel", help="Used as the column header in the CSV")
    c2.text_input("Unit", key="xunit", placeholder="e.g. h", help='Shown as "Time (h)"')
    c1, c2 = st.sidebar.columns([2, 1])
    c1.text_input("y-axis label", key="ylabel")
    c2.text_input("Unit", key="yunit", placeholder="e.g. Bq")

    st.sidebar.header("View")
    st.sidebar.select_slider("Zoom", options=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], key="zoom",
                             format_func=times_text,
                             help="Zoom in to place points more precisely")
    if ss.zoom > 1:
        st.sidebar.slider("Pan left / right", 0.0, 1.0, key="pan_x", step=0.01)
        st.sidebar.slider("Pan up / down", 0.0, 1.0, key="pan_y", step=0.01)
        # small preview with a red box around the part being shown
        x0, y0, cw, ch = compute_view(img.width, img.height, ss.zoom, ss.pan_x, ss.pan_y)
        thumb = img.copy()
        thumb.thumbnail((300, 300))
        k = thumb.width / img.width
        ImageDraw.Draw(thumb).rectangle(
            [x0 * k, y0 * k, (x0 + cw) * k, (y0 + ch) * k], outline="#ff2b2b", width=3)
        st.sidebar.image(thumb, caption="Visible region")
    st.sidebar.checkbox("Snap clicks to markers and ticks", key="snap",
                        help="Calibration clicks jump onto the nearest tick mark, and data "
                             "clicks move to the centre of the marker or line under them")
    st.sidebar.checkbox("Calibration check grid", key="show_grid",
                        help="Blue gridlines worked out from your calibration. If they sit "
                             "on the plot's own gridlines, the calibration is right.")
    st.sidebar.checkbox("Connect points with lines", key="connect")


# ---- Main page sections ----

def step_tracker(calibrated, n_pts):
    """Progress bar showing which of the four steps the user is on."""
    steps = [("Upload an image", True), ("Calibrate the axes", calibrated),
             ("Click the data", n_pts > 0), ("Download the CSV", False)]
    done = 0
    for name, finished in steps:
        if not finished:
            break
        done += 1
    st.progress(done / len(steps), text=f"Step {done + 1} of {len(steps)} · {steps[done][0]}")


def instruction(calibrated, cal_err):
    """What to do next, shown at the top of the right-hand panel."""
    if ss.mode != MODE_CAL and not calibrated:
        st.warning(f"Calibrate the axes first. {cal_err}")
    elif ss.mode == MODE_CAL:
        if ss.calib_next:
            st.info(f"Click the image to place **{ss.calib_next}**, {CAL_HELP[ss.calib_next]}.")
        elif calibrated:
            st.success("Calibration done. Check the values below, then start digitizing.")
        else:
            st.warning(cal_err or "")
    elif ss.mode == MODE_ADD:
        name = ss.datasets[ss.active]["name"]
        st.info(f"Click data points to add them to **{name}**. "
                "Use **+ New dataset** for another curve.")
    elif ss.mode == MODE_TRACE:
        st.info("Click right on a curve to trace it. Each curve you click goes into its own dataset.")
    else:
        st.info("Click a point on the image to delete it.")


def calibration_panel(calibrated, cal_err, data):
    st.subheader("Axis calibration")
    st.caption("Place two points on each axis at values you know (labelled ticks are best). "
               "Rotated or skewed scans are handled automatically.")
    st.button("Auto-detect axes", on_click=cb_auto_axes, args=(data,), width="stretch",
              help="Finds the axis lines and puts X1, X2, Y1, Y2 on the first and last "
                   "tick marks. You still type in the values.")
    for axis, keys in [("x", ["X1", "X2"]), ("y", ["Y1", "Y2"])]:
        with st.container(border=True):
            st.selectbox(f"{axis}-axis scale", AXIS_SCALES, key=f"{axis}scale",
                         help="How values are spaced along this axis. Most plots are linear; "
                              "log axes go 1, 10, 100...")
            for k in keys:
                c1, c2 = st.columns([1.1, 1])
                text = f"{k} ✓" if ss.calib[k] is not None else f"Set {k}"
                c1.button(text, key=f"pick_{k}",
                          type="primary" if ss.calib_next == k else "secondary",
                          on_click=cb_set, kwargs={"calib_next": k}, width="stretch",
                          help=f"Then click {CAL_HELP[k]}")
                with c2:
                    is_date = scale_of(k) == DATE
                    st.text_input(f"{k} value", key=f"{k}_val", label_visibility="collapsed",
                                  placeholder="e.g. 2024-03-01" if is_date else "e.g. 10^4",
                                  help="A date like 2024-03-01 or 2024-03-01 14:30" if is_date
                                  else "Any number: 250, -3.5, 1e4, 10^4, 2x10^-3, 1/3")
                    show_parsed(ss[f"{k}_val"], scale_of(k))

    if calibrated:
        tx, ty = axis_tilt(ss.calib)
        msg = f"x-axis tilt {tx:+.2f}° · y-axis tilt {ty:+.2f}°"
        if max(abs(tx), abs(ty)) > 0.3:
            msg += " · rotation corrected"
        st.caption(msg)
        if ss.show_grid:
            st.caption("Blue check grid should sit on the plot's gridlines.")
        st.button("Start digitizing →", type="primary", width="stretch",
                  on_click=cb_set, kwargs={"mode": MODE_ADD})
    elif cal_err and not ss.calib_next:
        st.error(cal_err)
    st.button("Reset calibration", on_click=cb_reset_calibration, width="stretch")


def clean_options():
    """(round to precision?, x step or None, y step or None) from the clean-up boxes."""
    steps = []
    for k in ["x_step", "y_step"]:
        try:
            step = parse_number(ss[k]) if ss[k].strip() else None
        except ValueError:
            step = None
        steps.append(step if step and step > 0 else None)
    return ss.round_prec, steps[0], steps[1]  # (ignored on date axes)


def cleaned(cal, pts):
    """Data values of pixel points, tidied with the clean-up options."""
    return clean_values(cal.to_data(pts), cal, *clean_options())


def titles():
    """Column headers for x and y, with units if given: "Time (h)"."""
    return axis_title(ss.xlabel or "x", ss.xunit), axis_title(ss.ylabel or "y", ss.yunit)


def show_parsed(text, scale):
    """Under a value box: what the typed text was read as (only if it isn't obvious)."""
    try:
        v = parse_value(text, scale)
    except ValueError:
        st.caption("⚠ can't read this" if scale != DATE else "⚠ type a date, e.g. 2024-03-01")
        return
    shown = fmt_value(v, scale)
    if shown != str(text).strip():
        st.caption(f"= {shown}")


def dataset_label(i):
    """Radio-button text for dataset i, e.g. "🔴 Dataset 1 · 5 points"."""
    ds = ss.datasets[i]
    n = len(ds["points"])
    return f"{colour_dot(ds['color'])} {ds['name']} · {n} point{'' if n == 1 else 's'}"


def dataset_list():
    """Pick the active dataset (new clicks go into it)."""
    ss.active_radio = ss.active  # ss.active is the real value; copy it in before drawing
    st.radio("Active dataset (new clicks go here)", list(range(len(ss.datasets))),
             format_func=dataset_label, key="active_radio",
             on_change=cb_copy, args=("active_radio", "active"))


def trace_settings():
    with st.container(border=True):
        st.markdown("**Auto-trace settings**")
        st.slider("Points per curve", 5, 100, key="trace_n",
                  help="How many evenly spaced points to take along each traced curve")
        st.slider("Colour tolerance", 20, 150, key="trace_tol",
                  help="How different a pixel's colour can be from the curve's and still count. "
                       "Raise it for faded or blurry scans, lower it if it picks up other lines.")


def advanced_tools(ds, cal):
    """Reorder, sort, reverse and merge (hidden behind a switch to keep things simple)."""
    i = ss.active
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.button("↑ Move up", on_click=cb_move_dataset, args=(i, -1), width="stretch",
                  disabled=i == 0, help="Earlier in the list and in the CSV")
        c2.button("↓ Move down", on_click=cb_move_dataset, args=(i, 1), width="stretch",
                  disabled=i == len(ss.datasets) - 1)
        c1, c2 = st.columns(2)
        c1.button("Sort by x", on_click=cb_sort_dataset, args=(i,), width="stretch",
                  disabled=cal is None or len(ds["points"]) < 2,
                  help="Put this dataset's points in order of increasing x")
        c2.button("Reverse", on_click=cb_reverse_dataset, args=(i,), width="stretch",
                  disabled=len(ds["points"]) < 2, help="Flip the order of this dataset's points")
        others = [d for d in ss.datasets if d["uid"] != ds["uid"]]
        if others:
            names = {d["uid"]: d["name"] for d in others}
            uid = st.selectbox("Merge another dataset into this one", list(names),
                               format_func=names.get, key=f"merge_{ds['uid']}")
            st.button(f"Merge into {ds['name']}", on_click=cb_merge_dataset,
                      args=(i, uid), width="stretch")


def datasets_panel(cal):
    st.subheader("Datasets")
    ss.active = min(ss.active, len(ss.datasets) - 1)
    dataset_list()
    st.button("+ New dataset", on_click=cb_add_dataset, width="stretch",
              help="Add another dataset, e.g. for a second curve")

    ds = ss.datasets[ss.active]
    st.markdown("**Edit the selected dataset**")
    c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
    name = c1.text_input("Name", value=ds["name"], key=f"name_{ds['uid']}").strip()
    ds["name"] = name or f"Dataset {ss.active + 1}"
    ds["color"] = c2.color_picker("Colour", value=ds["color"], key=f"color_{ds['uid']}")

    b1, b2 = st.columns(2)
    no_points = not ds["points"]
    b1.button("↶ Last", on_click=cb_remove_points, args=(ss.active, True), width="stretch",
              disabled=no_points, help="Remove the last point of this dataset")
    b2.button("Clear", on_click=cb_remove_points, args=(ss.active, False), width="stretch",
              disabled=no_points, help="Remove all points of this dataset")
    with st.popover("Delete dataset", width="stretch"):
        st.write(f"Delete **{ds['name']}** and its {len(ds['points'])} points?")
        st.button("Yes, delete", type="primary", on_click=cb_delete_dataset, args=(ss.active,))
    st.toggle("Advanced dataset tools", key="adv_tools",
              help="Reorder, sort, reverse or merge datasets")
    if ss.adv_tools:
        advanced_tools(ds, cal)

    st.markdown("**Points** (edit a value to move its mark)")
    if cal is None:
        st.caption("Calibrate the axes to see values.")
        return
    pix = np.asarray(ds["points"], dtype=float).reshape(-1, 2)
    vals = cleaned(cal, pix)
    df = pd.DataFrame()
    config = {}
    for col, (name, title, scale) in enumerate(zip(["x", "y"], titles(), cal.scales)):
        if scale == DATE:  # dates are shown and typed as text
            df[name] = [fmt_value(v, DATE) for v in vals[:, col]]
            config[name] = st.column_config.TextColumn(title)
        else:
            df[name] = [float(fmt(v)) for v in vals[:, col]]
            config[name] = st.column_config.NumberColumn(title, format="%.6g")
    edited = st.data_editor(df, key=f"ed_{ds['uid']}_{ss.editor_ver}", num_rows="dynamic",
                            width="stretch", height=min(38 + 35 * (len(df) + 1), 360),
                            column_config=config)
    apply_table_edits(ds, df, edited, pix, cal)


def table_numbers(frame, scales):
    """The table's x and y columns as numbers (dates back to seconds; blanks -> nan)."""
    out = np.full((len(frame), 2), np.nan)
    for col, (name, scale) in enumerate(zip(["x", "y"], scales)):
        for i, v in enumerate(frame[name]):
            if v is None or str(v).strip() in ("", "nan", "None"):
                continue
            out[i, col] = parse_date(v) if scale == DATE else float(v)
    return out


def apply_table_edits(ds, df, edited, pix, cal):
    """Move, add or remove points to match what was typed into the table."""
    try:
        old, new = table_numbers(df, cal.scales), table_numbers(edited, cal.scales)
    except ValueError as e:
        st.error(str(e))
        return
    if edited.index.equals(df.index) and np.allclose(old, new, equal_nan=True):
        return  # nothing changed
    if np.isnan(new).any():
        st.caption("Enter both x and y for the new row.")
        return

    new_pts = []
    try:
        for row, idx in enumerate(edited.index):
            if idx in df.index and np.allclose(new[row], old[df.index.get_loc(idx)]):
                # row not changed: keep the exact clicked pixel, not the rounded value
                new_pts.append(list(pix[df.index.get_loc(idx)]))
            else:
                new_pts.append(list(cal.to_pixel([new[row]])[0]))
    except ValueError as e:
        st.error(str(e))
        return
    snapshot()
    ds["points"] = new_pts
    ss.editor_ver += 1
    st.rerun()


def unique_names(names):
    """st.dataframe needs unique column names, so number any repeats: "a", "a (2)"."""
    seen = {}
    result = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return result


def clean_up_card(cal):
    """Options that stop mouse wobble showing up as extra digits (0.0124 instead of 0)."""
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        c1.checkbox("Round to image precision", key="round_prec",
                    help="Drops digits finer than one pixel of the image, "
                         f"which are just mouse wobble. One pixel here is {precision_text(cal)}.")
        c1.caption(f"One pixel = {precision_text(cal)}")
        for col, k, name in [(c2, "x_step", "x"), (c3, "y_step", "y")]:
            col.text_input(f"Snap {name} to multiples of", key=k, placeholder="off",
                           help=f"For data taken at round {name} values (e.g. 1 or 0.5). "
                                "Leave blank to turn off.")
            if (ss.xscale if name == "x" else ss.yscale) == DATE:
                col.caption("not used on a date axis")
            elif ss[k].strip():
                try:
                    parse_number(ss[k])
                except ValueError as e:
                    col.caption(f"⚠ {e}")


def results_section(cal, file_stem):
    st.divider()
    if cal is None:
        st.subheader("Results")
        st.caption("Calibrate the axes and click some points to see the results here.")
        return

    export = []
    for ds in ss.datasets:
        data = cleaned(cal, ds["points"])
        if ss.sort_x:
            data = data[np.argsort(data[:, 0], kind="stable")]
        export.append({"name": ds["name"], "color": ds["color"],
                       "data": [list(r) for r in data]})
    xtitle, ytitle = titles()
    rows = csv_rows(export, xtitle, ytitle, ss.xscale, ss.yscale)
    filled = [e for e in export if e["data"]]
    total = sum([len(e["data"]) for e in filled])
    st.subheader("Results")
    st.caption(f"{len(filled)} datasets · {total} points")
    clean_up_card(cal)

    c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="center")
    c1.download_button("Download CSV", rows_to_text(rows).encode("utf-8-sig"),
                       file_name=f"{file_stem or 'digitized'}_data.csv", mime="text/csv",
                       type="primary", disabled=not filled, width="stretch")
    c2.checkbox("Sort each dataset by x", key="sort_x")
    names = [e["name"] for e in filled]
    if len(set(names)) < len(names):
        c3.warning("Two datasets share a name. Rename one so the CSV is clear.")
    if not filled:
        st.caption("No points yet.")
        return

    table, plot = st.columns([3, 2], gap="large")
    with table:
        # preview table, same layout as the CSV with "name · label" headers
        st.markdown("**CSV preview**")
        header = [f"{e['name']} · {lab}" for e in filled for lab in (xtitle, ytitle)]
        st.dataframe(pd.DataFrame(rows[2:], columns=unique_names(header)), width="stretch",
                     hide_index=True, height=min(38 + 35 * (len(rows) - 2), 420))
    with plot:
        st.markdown("**Check plot** (compare with the original)")
        st.image(cached_replot(filled, xtitle, ytitle, ss.xscale, ss.yscale))


def welcome():
    st.title("Plot Digitizer")
    st.subheader("Turn any graph into data you can use")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("**How it works**")
        steps = [("Upload", "PNG, JPG, GIF or TIFF · colour or greyscale"),
                 ("Calibrate", "Two known points per axis · linear or log · crooked scans OK"),
                 ("Digitize", "Click the data · one colour per dataset"),
                 ("Export", "CSV with dataset names and axis labels")]
        for i, (title, text) in enumerate(steps, 1):
            with st.container(border=True):
                st.markdown(f"**0{i} · {title}**")
                st.caption(text)
        st.button("Try a sample plot", type="primary", width="stretch",
                  on_click=cb_set, kwargs={"use_sample": True})
        st.caption("Or upload your own image in the sidebar.")


def trace_click(img, data, cal, px, py):
    """Trace the clicked curve into the active dataset (or a new one if it has points)."""
    if cal is None:
        st.toast("Calibrate the axes first.", icon="⚠️")
        return
    try:
        ink = cached_ink(data, ss.rotation)
        colour, pts = trace_curve(img, ink, (px, py), cal, ss.calib, ss.trace_n, ss.trace_tol)
    except ValueError as e:
        st.toast(str(e), icon="⚠️")
        return
    snapshot()
    if ss.datasets[ss.active]["points"]:
        ss.datasets.append(new_dataset())
        ss.active = len(ss.datasets) - 1
    ds = ss.datasets[ss.active]
    ds["points"] = [[float(x), float(y)] for x, y in pts]
    ss.editor_ver += 1
    st.toast(f"Traced {len(pts)} points into {ds['name']}.")


def image_panel(img, img_hash, data, cal):
    view = compute_view(img.width, img.height, ss.zoom, ss.pan_x, ss.pan_y)
    highlight = ss.calib_next if ss.mode == MODE_CAL else None
    grid = calibration_grid(cal, img.size, ss.calib) if cal is not None and ss.show_grid else None
    # drawn sharp at RENDER_W, then the browser scales it to fit the column
    disp = render_view(img, view, RENDER_W, ss.calib, ss.datasets, ss.active, highlight,
                       ss.connect, grid)
    value = streamlit_image_coordinates(disp, key=f"img_{img_hash}", width="stretch",
                                        png_compression_level=1, cursor="crosshair")
    # the component keeps returning its last click, so only act on new ones
    if value and value != ss.last_click:
        ss.last_click = value
        x0, y0, cw, ch = view
        # the click is in displayed pixels; scale it back to the original image
        px = x0 + value["x"] * cw / value["width"]
        py = y0 + value["y"] * ch / value["height"]
        if ss.mode == MODE_TRACE:
            trace_click(img, data, cal, px, py)
            st.rerun()
        if ss.mode == MODE_CAL and ss.snap:
            px, py = nearest_tick(cached_ticks(data, ss.rotation), px, py,
                                  radius=max(8.0, 0.015 * img.width))
        if ss.mode == MODE_ADD and ss.snap:
            px, py = snap_to_marker(cached_ink(data, ss.rotation), px, py,
                                    radius=max(5.0, 0.012 * img.width))
        # delete tolerance is about 1.5% of the visible width
        handle_click(px, py, 0.015 * cw)
        st.rerun()


def main():
    st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")
    init_state()

    data, stem = sidebar_image()
    if data is None:
        welcome()
        return
    st.title("Plot Digitizer")

    img_hash = hash(data)  # same bytes -> same number, so we can tell when the image changes
    if ss.img_hash != img_hash:  # new image, so start fresh
        ss.img_hash = img_hash
        reset_work()
    try:
        info = cached_decode(data)[1]
        tilt = cached_tilt(data)
    except Exception as e:  # corrupt or unsupported file
        st.error(f"Could not read this image: {e}")
        return
    # work on a straightened copy if the scan is crooked (and the user wants that)
    angle = -tilt if ss.straighten and abs(tilt) >= 0.2 else 0.0
    if angle != ss.rotation:
        change_rotation(data, angle)
    img = cached_rotate(data, angle)
    sidebar_settings(img, info, tilt)

    cal, cal_err = current_calibration()
    calibrated = cal is not None
    n_pts = sum([len(ds["points"]) for ds in ss.datasets])

    top1, top2 = st.columns([5, 1], vertical_alignment="bottom")
    with top1:
        step_tracker(calibrated, n_pts)
    top2.button("↶ Undo", on_click=undo, disabled=not ss.history, width="stretch",
                help="Undo the last change (points, datasets or calibration)")
    # ss.mode is the real value; copy it into the radio before drawing it
    ss.mode_radio = ss.mode
    st.radio("Mode", MODES, key="mode_radio", horizontal=True, label_visibility="collapsed",
             on_change=cb_copy, args=("mode_radio", "mode"))

    left, right = st.columns([5, 3], gap="medium")
    with left:
        image_panel(img, img_hash, data, cal)
    with right:
        instruction(calibrated, cal_err)
        if ss.mode == MODE_CAL:
            calibration_panel(calibrated, cal_err, data)
        else:
            if ss.mode == MODE_TRACE:
                trace_settings()
            datasets_panel(cal)

    results_section(cal, stem)


if __name__ == "__main__":
    main()
