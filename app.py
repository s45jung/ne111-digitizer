"""NE111 Plot Digitizer.

A single-page Streamlit app for extracting x-y data from an image of a graph.

Workflow
--------
1. Upload an image of a plot (png, jpg, gif, tiff, bmp, webp; colour or greyscale).
2. Calibrate: click two known points on the x-axis (X1, X2) and two on the
   y-axis (Y1, Y2), then type their values. Linear and log axes are supported,
   and the calibration is a full affine fit, so rotated or skewed scans work.
3. Digitize: click data points. Each dataset has its own name and colour.
4. Download the CSV (row 1 = dataset names, row 2 = axis labels, then values).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # lets the pure functions below be imported for testing
    streamlit_image_coordinates = None

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a5c8",
    "#f032e6", "#9a6324", "#469990", "#800000", "#808000", "#000075",
]
CAL_KEYS = ["X1", "X2", "Y1", "Y2"]
CAL_COLORS = {"X1": "#d62728", "X2": "#d62728", "Y1": "#1f77b4", "Y2": "#1f77b4"}
CAL_HELP = {
    "X1": "a point on the **x-axis** whose value you know (e.g. the first labelled tick)",
    "X2": "a second **x-axis** point, as far from X1 as possible",
    "Y1": "a point on the **y-axis** whose value you know (e.g. the lowest labelled tick)",
    "Y2": "a second **y-axis** point, as far from Y1 as possible",
}
MODE_CAL, MODE_ADD, MODE_DEL = "📐 Calibrate axes", "➕ Add points", "✖️ Delete points"
MODES = [MODE_CAL, MODE_ADD, MODE_DEL]
IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp", "webp"]
# Widget values that must survive while their widget is not on screen.
PERSIST_KEYS = ["xlabel", "ylabel", "xlog", "ylog", "X1_val", "X2_val", "Y1_val",
                "Y2_val", "zoom", "pan_x", "pan_y", "disp_w", "sort_x", "connect"]
MAX_HISTORY = 100


# --------------------------------------------------------------------------
# Image handling (pure functions)
# --------------------------------------------------------------------------

def to_rgb(im: Image.Image) -> tuple[Image.Image, dict]:
    """Convert any PIL image (palette, 16-bit, alpha, CMYK, animated...) to 8-bit RGB."""
    info = {"format": im.format or "image", "mode": im.mode,
            "frames": getattr(im, "n_frames", 1)}
    if info["frames"] > 1:
        im.seek(0)  # first frame of animated GIF / multi-page TIFF
    try:
        im = ImageOps.exif_transpose(im)  # phone photos store rotation in EXIF
    except Exception:
        pass
    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        arr = np.asarray(im, dtype=np.float64)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        im = Image.fromarray(np.nan_to_num(arr).astype(np.uint8), "L")
    if im.mode == "1":
        im = im.convert("L")
    if im.mode in ("P", "PA"):
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im.convert("RGBA"))
        im = bg
    rgb = im.convert("RGB")
    info["greyscale"] = is_greyscale(rgb)
    info["size"] = rgb.size
    return rgb, info


def is_greyscale(rgb: Image.Image) -> bool:
    small = np.asarray(rgb.resize((min(200, rgb.width), min(200, rgb.height))), dtype=np.int16)
    return bool(np.max(np.abs(small[..., 0] - small[..., 1])) < 10
                and np.max(np.abs(small[..., 1] - small[..., 2])) < 10)


def decode_image(data: bytes) -> tuple[Image.Image, dict]:
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return to_rgb(im)


def make_sample_plot() -> bytes:
    """A semi-log plot with two curves, scanned slightly crooked (rotated 3°)."""
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
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    im = im.rotate(3, expand=True, fillcolor="white", resample=Image.BICUBIC)
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


# --------------------------------------------------------------------------
# Calibration (pure functions)
# --------------------------------------------------------------------------

def _axis_value(v: float, log: bool, name: str) -> float:
    if log:
        if v is None or v <= 0:
            raise ValueError(f"{name} must be greater than 0 on a logarithmic axis.")
        return math.log10(v)
    return float(v)


def solve_calibration(px: dict, vals: dict, xlog: bool, ylog: bool):
    """Return (A, b) such that [X', Y'] = A @ [col, row] + b.

    X' is x (or log10 x) and Y' is y (or log10 y). The x-value is fixed by X1
    and X2 and is constant along the y-axis direction (Y1 -> Y2); likewise for y.
    That is a general affine map, so rotation, skew and unequal scaling are all
    handled, and the four points do not need to share an origin.
    """
    missing = [k for k in CAL_KEYS if px.get(k) is None]
    if missing:
        raise ValueError("Place calibration point(s): " + ", ".join(missing) + ".")
    X1, X2, Y1, Y2 = (np.asarray(px[k], dtype=float) for k in CAL_KEYS)
    x1 = _axis_value(vals["X1"], xlog, "X1")
    x2 = _axis_value(vals["X2"], xlog, "X2")
    y1 = _axis_value(vals["Y1"], ylog, "Y1")
    y2 = _axis_value(vals["Y2"], ylog, "Y2")
    if x1 == x2:
        raise ValueError("X1 and X2 must have different values.")
    if y1 == y2:
        raise ValueError("Y1 and Y2 must have different values.")
    if np.allclose(X1, X2):
        raise ValueError("X1 and X2 are at the same spot. Place them further apart.")
    if np.allclose(Y1, Y2):
        raise ValueError("Y1 and Y2 are at the same spot. Place them further apart.")
    dx, dy = X2 - X1, Y2 - Y1
    try:
        a, b, c = np.linalg.solve(
            [[X1[0], X1[1], 1], [X2[0], X2[1], 1], [dy[0], dy[1], 0]], [x1, x2, 0])
        d, e, f = np.linalg.solve(
            [[Y1[0], Y1[1], 1], [Y2[0], Y2[1], 1], [dx[0], dx[1], 0]], [y1, y2, 0])
    except np.linalg.LinAlgError:
        raise ValueError("The x-axis and y-axis points are parallel. "
                         "Check that X1/X2 are on the x-axis and Y1/Y2 on the y-axis.")
    A = np.array([[a, b], [d, e]])
    if abs(np.linalg.det(A)) < 1e-15:
        raise ValueError("Calibration is degenerate. Re-place the points.")
    return A, np.array([c, f])


def pixel_to_data(pts, A, off, xlog, ylog) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    lin = pts @ A.T + off
    if xlog:
        lin[:, 0] = 10.0 ** lin[:, 0]
    if ylog:
        lin[:, 1] = 10.0 ** lin[:, 1]
    return lin


def data_to_pixel(vals, A, off, xlog, ylog) -> np.ndarray:
    v = np.asarray(vals, dtype=float).reshape(-1, 2).copy()
    if xlog:
        if np.any(v[:, 0] <= 0):
            raise ValueError("x values must be greater than 0 on a log axis.")
        v[:, 0] = np.log10(v[:, 0])
    if ylog:
        if np.any(v[:, 1] <= 0):
            raise ValueError("y values must be greater than 0 on a log axis.")
        v[:, 1] = np.log10(v[:, 1])
    return (v - off) @ np.linalg.inv(A).T


def axis_tilt(px: dict) -> tuple[float | None, float | None]:
    """Tilt (degrees) of the x-axis from horizontal and the y-axis from vertical."""
    def tilt(p, q, vertical):
        if p is None or q is None:
            return None
        dx, dy = q[0] - p[0], q[1] - p[1]
        # counter-clockwise positive for both axes (image rows grow downward)
        ang = math.degrees(math.atan2(-dx, -dy) if vertical else math.atan2(-dy, dx))
        return (ang + 90) % 180 - 90  # fold into [-90, 90)
    return tilt(px.get("X1"), px.get("X2"), False), tilt(px.get("Y1"), px.get("Y2"), True)


# --------------------------------------------------------------------------
# CSV export (pure function)
# --------------------------------------------------------------------------

def fmt(v: float) -> str:
    return f"{v:.6g}"


def build_csv(datasets: list[dict], xlabel: str, ylabel: str) -> str:
    """Row 1: dataset names. Row 2: x/y labels. Then x, y pairs side by side."""
    cols = [ds for ds in datasets if len(ds["data"])]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow([c for ds in cols for c in (ds["name"], "")])
    w.writerow([c for _ in cols for c in (xlabel, ylabel)])
    n = max((len(ds["data"]) for ds in cols), default=0)
    for i in range(n):
        row = []
        for ds in cols:
            if i < len(ds["data"]):
                row += [fmt(ds["data"][i][0]), fmt(ds["data"][i][1])]
            else:
                row += ["", ""]
        w.writerow(row)
    return out.getvalue()


# --------------------------------------------------------------------------
# Drawing (pure functions)
# --------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def compute_view(W: int, H: int, zoom: float, pan_x: float, pan_y: float):
    """Visible region (x0, y0, w, h) of the original image, in original pixels."""
    cw, ch = W / zoom, H / zoom
    x0 = min(max(pan_x * W - cw / 2, 0.0), W - cw)
    y0 = min(max(pan_y * H - ch / 2, 0.0), H - ch)
    return x0, y0, cw, ch


def render_view(img, view, disp_w, calib, datasets, active, highlight=None, connect=False):
    x0, y0, cw, ch = view
    s = disp_w / cw
    disp_h = max(1, round(ch * s))
    resample = Image.LANCZOS if s < 1 else Image.BICUBIC
    base = img.resize((disp_w, disp_h), resample, box=(x0, y0, x0 + cw, y0 + ch)).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    to_d = lambda p: ((p[0] - x0) * s, (p[1] - y0) * s)  # noqa: E731
    font = _font(15)

    # Calibration axes and markers
    for p, q, col in (("X1", "X2", CAL_COLORS["X1"]), ("Y1", "Y2", CAL_COLORS["Y1"])):
        if calib.get(p) and calib.get(q):
            d.line([to_d(calib[p]), to_d(calib[q])], fill=col + "99", width=2)
    for k in CAL_KEYS:
        if calib.get(k) is None:
            continue
        cx, cy = to_d(calib[k])
        col = CAL_COLORS[k]
        r = 11 if k == highlight else 8
        d.line([(cx - r, cy), (cx + r, cy)], fill=col, width=2)
        d.line([(cx, cy - r), (cx, cy + r)], fill=col, width=2)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)
        d.text((cx + r + 3, cy - r - 12), k, fill=col, font=font,
               stroke_width=3, stroke_fill="white")

    # Data points, active dataset drawn last (on top) and larger
    order = [i for i in range(len(datasets)) if i != active] + [active]
    for i in order:
        if i >= len(datasets):
            continue
        ds = datasets[i]
        pts = [to_d(p) for p in ds["points"]]
        if connect and len(pts) > 1:
            d.line(pts, fill=ds["color"] + "aa", width=2)
        r = 6 if i == active else 4.5
        for cx, cy in pts:
            d.ellipse([cx - r - 1.5, cy - r - 1.5, cx + r + 1.5, cy + r + 1.5], fill="#000000aa")
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ds["color"], outline="white", width=2)
    return Image.alpha_composite(base, layer).convert("RGB")


def nearest_point(datasets, px, py, tol):
    best, best_d = None, tol
    for i, ds in enumerate(datasets):
        for j, (qx, qy) in enumerate(ds["points"]):
            dist = math.hypot(qx - px, qy - py)
            if dist <= best_d:
                best, best_d = (i, j), dist
    return best


# --------------------------------------------------------------------------
# Session state helpers
# --------------------------------------------------------------------------

ss = st.session_state


def new_dataset() -> dict:
    ss.ds_counter = ss.get("ds_counter", 0) + 1
    n = ss.ds_counter
    used = {ds["color"] for ds in ss.get("datasets", [])}
    color = next((c for c in PALETTE if c not in used), PALETTE[(n - 1) % len(PALETTE)])
    return {"uid": uuid.uuid4().hex[:8], "name": f"Dataset {n}", "color": color, "points": []}


def reset_work():
    ss.calib = {k: None for k in CAL_KEYS}
    ss.calib_next = "X1"
    ss.ds_counter = 0
    ss.datasets = []
    ss.datasets = [new_dataset()]
    ss.active = 0
    ss.history = []
    ss.last_click = None
    ss.editor_ver = 0
    ss.mode = MODE_CAL
    ss.zoom, ss.pan_x, ss.pan_y = 1.0, 0.5, 0.5


def init_state():
    for k in PERSIST_KEYS:  # keep values alive when their widget is hidden
        if k in ss:
            ss[k] = ss[k]
    defaults = {"xlabel": "x", "ylabel": "y", "xlog": False, "ylog": False,
                "X1_val": 0.0, "X2_val": 10.0, "Y1_val": 0.0, "Y2_val": 10.0,
                "zoom": 1.0, "pan_x": 0.5, "pan_y": 0.5, "disp_w": 760,
                "sort_x": False, "connect": False, "img_hash": None, "use_sample": False}
    for k, v in defaults.items():
        ss.setdefault(k, v)
    if "datasets" not in ss:
        reset_work()


def snapshot():
    ss.history.append(copy.deepcopy(
        {"calib": ss.calib, "datasets": ss.datasets, "active": ss.active,
         "ds_counter": ss.ds_counter}))
    del ss.history[:-MAX_HISTORY]


def undo():
    if ss.history:
        snap = ss.history.pop()
        ss.calib, ss.datasets = snap["calib"], snap["datasets"]
        ss.active, ss.ds_counter = snap["active"], snap["ds_counter"]
        ss.editor_ver += 1


def calibration():
    """(A, off, None) if calibrated, else (None, None, error message)."""
    try:
        A, off = solve_calibration(
            ss.calib, {k: ss[f"{k}_val"] for k in CAL_KEYS}, ss.xlog, ss.ylog)
        return A, off, None
    except ValueError as e:
        return None, None, str(e)


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------

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


def cb_remove_last(i):
    if ss.datasets[i]["points"]:
        snapshot()
        ss.datasets[i]["points"].pop()
        ss.editor_ver += 1


def cb_clear(i):
    if ss.datasets[i]["points"]:
        snapshot()
        ss.datasets[i]["points"] = []
        ss.editor_ver += 1


def cb_reset_calibration():
    snapshot()
    ss.calib = {k: None for k in CAL_KEYS}
    ss.calib_next = "X1"
    ss.mode = MODE_CAL


def handle_click(px: float, py: float, tol: float):
    if ss.mode == MODE_CAL:
        target = ss.calib_next
        if target is None:
            st.toast("All four calibration points are placed. Pick one on the right to move it.")
            return
        snapshot()
        ss.calib[target] = (px, py)
        unset = [k for k in CAL_KEYS if ss.calib[k] is None]
        after = CAL_KEYS[CAL_KEYS.index(target) + 1:]
        ss.calib_next = next((k for k in after if k in unset), unset[0] if unset else None)
    elif ss.mode == MODE_ADD:
        A, _, err = calibration()
        if A is None:
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


# --------------------------------------------------------------------------
# UI sections
# --------------------------------------------------------------------------

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
.step {display:inline-block; padding:.25rem .75rem; margin-right:.4rem; border-radius:999px;
       font-size:.9rem; border:1px solid rgba(128,128,128,.35);}
.step.done {background:rgba(33,195,84,.15); border-color:rgba(33,195,84,.6);}
.step.now {background:rgba(28,131,225,.15); border-color:rgba(28,131,225,.7); font-weight:600;}
.swatch {display:inline-block; width:.9rem; height:.9rem; border-radius:50%;
         border:2px solid white; box-shadow:0 0 0 1px #0006; vertical-align:-2px; margin-right:.35rem;}
.legend span.item {margin-right:1rem; white-space:nowrap;}
</style>
"""


def sidebar_image() -> tuple[bytes | None, str]:
    st.sidebar.header("1 · Image")
    up = st.sidebar.file_uploader(
        "Upload a plot image", type=IMAGE_TYPES,
        help="PNG, JPG, GIF, TIFF, BMP or WebP. Colour or greyscale.")
    if up is not None:
        return up.getvalue(), Path(up.name).stem
    c1, c2 = st.sidebar.columns(2)
    c1.button("Try a sample plot", on_click=cb_set, kwargs={"use_sample": True},
              help="A semi-log plot scanned 3° crooked")
    if ss.use_sample:
        c2.button("Close sample", on_click=cb_set, kwargs={"use_sample": False})
        return cached_sample(), "sample_plot"
    return None, ""


@st.cache_data(show_spinner=False)
def cached_sample() -> bytes:
    return make_sample_plot()


@st.cache_data(show_spinner=False, max_entries=4)
def cached_decode(data: bytes):
    return decode_image(data)


def sidebar_settings(img: Image.Image, info: dict):
    kind = "greyscale" if info["greyscale"] else "colour"
    frames = f" · first of {info['frames']} frames" if info["frames"] > 1 else ""
    st.sidebar.caption(f"{info['format']} · {info['size'][0]}×{info['size'][1]} px · {kind}{frames}")

    st.sidebar.header("2 · Axis labels")
    st.sidebar.text_input("x-axis label", key="xlabel", help="Used as the column header in the CSV")
    st.sidebar.text_input("y-axis label", key="ylabel")

    st.sidebar.header("3 · View")
    st.sidebar.select_slider("Zoom", options=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], key="zoom",
                             format_func=lambda z: f"{z:g}×",
                             help="Zoom in to place points more precisely")
    if ss.zoom > 1:
        st.sidebar.slider("Pan left ↔ right", 0.0, 1.0, key="pan_x", step=0.01)
        st.sidebar.slider("Pan up ↕ down", 0.0, 1.0, key="pan_y", step=0.01)
        view = compute_view(img.width, img.height, ss.zoom, ss.pan_x, ss.pan_y)
        thumb = img.copy()
        thumb.thumbnail((300, 300))
        k = thumb.width / img.width
        x0, y0, cw, ch = view
        ImageDraw.Draw(thumb).rectangle(
            [x0 * k, y0 * k, (x0 + cw) * k, (y0 + ch) * k], outline="#ff2b2b", width=3)
        st.sidebar.image(thumb, caption="Visible region")
    st.sidebar.slider("Display width (px)", 400, 1600, key="disp_w", step=20,
                      help="Size of the image on screen")
    st.sidebar.checkbox("Connect points with lines", key="connect")


def step_tracker(have_img: bool, calibrated: bool, n_pts: int):
    steps = [("Upload image", have_img), ("Calibrate axes", calibrated),
             ("Click data points", n_pts > 0), ("Download CSV", False)]
    html, current_found = [], False
    for i, (name, done) in enumerate(steps, 1):
        cls = "done" if done else ("now" if not current_found else "")
        if not done:
            current_found = True
        html.append(f'<span class="step {cls}">{"✓" if done else i} {name}</span>')
    st.markdown("".join(html), unsafe_allow_html=True)


def instruction(calibrated: bool, cal_err: str | None):
    if ss.mode == MODE_CAL:
        if ss.calib_next:
            st.info(f"**Click the image to place {ss.calib_next}**: {CAL_HELP[ss.calib_next]}.",
                    icon="📐")
        elif calibrated:
            st.success("Calibration done. Check the values on the right, then start digitizing.",
                       icon="✅")
        else:
            st.warning(cal_err, icon="⚠️")
    elif ss.mode == MODE_ADD:
        if not calibrated:
            st.warning(f"Calibrate the axes first. {cal_err}", icon="⚠️")
        else:
            ds = ss.datasets[ss.active]
            st.info(f"**Click data points** to add them to "
                    f"<span class='swatch' style='background:{ds['color']}'></span>"
                    f"**{ds['name']}**.", icon="➕")
    else:
        st.info("**Click a point** to delete it.", icon="✖️")


def calibration_panel(calibrated: bool, cal_err: str | None):
    st.subheader("Axis calibration")
    st.caption("Place two points on each axis at values you know (labelled ticks are best). "
               "Rotated or skewed scans are handled automatically.")
    for axis, keys in (("x", ("X1", "X2")), ("y", ("Y1", "Y2"))):
        with st.container(border=True):
            st.toggle(f"Logarithmic {axis}-axis", key=f"{axis}log")
            for k in keys:
                c1, c2 = st.columns([1.1, 1])
                placed = ss.calib[k] is not None
                is_next = ss.calib_next == k
                label = f"{'✓' if placed else '○'} {'Move' if placed else 'Place'} {k}"
                c1.button(label, key=f"pick_{k}", type="primary" if is_next else "secondary",
                          on_click=cb_set, kwargs={"calib_next": k}, width="stretch",
                          help=f"Then click {CAL_HELP[k]}")
                c2.number_input(f"{k} value", key=f"{k}_val", format="%g",
                                label_visibility="collapsed")
    tx, ty = axis_tilt(ss.calib)
    if calibrated:
        tilt = max(abs(tx or 0), abs(ty or 0))
        msg = f"x-axis tilt {tx:+.2f}°, y-axis tilt {ty:+.2f}°"
        if tilt > 0.3:
            msg += " (rotation corrected)"
        st.caption("📏 " + msg)
        st.button("Start digitizing →", type="primary", width="stretch",
                  on_click=cb_set, kwargs={"mode": MODE_ADD})
    elif cal_err and not ss.calib_next:
        st.error(cal_err)
    st.button("Reset calibration", on_click=cb_reset_calibration)


def datasets_panel(A, off):
    st.subheader("Datasets")
    legend = "".join(
        f"<span class='item'><span class='swatch' style='background:{ds['color']}'></span>"
        f"{ds['name']} ({len(ds['points'])})</span>" for ds in ss.datasets)
    st.markdown(f"<div class='legend'>{legend}</div>", unsafe_allow_html=True)

    names = [ds["name"] for ds in ss.datasets]
    ss.active = min(ss.active, len(ss.datasets) - 1)
    ss.active_radio = ss.active
    c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
    c1.selectbox("Active dataset (new clicks go here)", range(len(names)), key="active_radio",
                 format_func=lambda i: names[i], on_change=cb_copy,
                 args=("active_radio", "active"))
    c2.button("➕ New", on_click=cb_add_dataset, width="stretch", help="Add another dataset")

    ds = ss.datasets[ss.active]
    c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
    ds["name"] = c1.text_input("Name", value=ds["name"], key=f"name_{ds['uid']}").strip() \
        or f"Dataset {ss.active + 1}"
    ds["color"] = c2.color_picker("Colour", value=ds["color"], key=f"color_{ds['uid']}")

    b1, b2, b3 = st.columns(3)
    b1.button("↶ Last point", on_click=cb_remove_last, args=(ss.active,), width="stretch",
              disabled=not ds["points"], help="Remove the last point of this dataset")
    b2.button("Clear", on_click=cb_clear, args=(ss.active,), width="stretch",
              disabled=not ds["points"], help="Remove all points of this dataset")
    with b3.popover("🗑 Delete", width="stretch"):
        st.write(f"Delete **{ds['name']}** and its {len(ds['points'])} points?")
        st.button("Yes, delete", type="primary", on_click=cb_delete_dataset, args=(ss.active,))

    st.markdown("**Points** · edit values or delete rows; the marks on the image update.")
    if A is None:
        st.caption("Calibrate the axes to see values.")
        return
    pix = np.asarray(ds["points"], dtype=float).reshape(-1, 2)
    vals = pixel_to_data(pix, A, off, ss.xlog, ss.ylog) if len(pix) else np.empty((0, 2))
    df = pd.DataFrame({"x": [float(fmt(v)) for v in vals[:, 0]],
                       "y": [float(fmt(v)) for v in vals[:, 1]]})
    edited = st.data_editor(
        df, key=f"ed_{ds['uid']}_{ss.editor_ver}", num_rows="dynamic", width="stretch",
        height=min(38 + 35 * (len(df) + 1), 360),
        column_config={"x": st.column_config.NumberColumn(ss.xlabel or "x", format="%.6g"),
                       "y": st.column_config.NumberColumn(ss.ylabel or "y", format="%.6g")})
    apply_table_edits(ds, df, edited, pix, A, off)


def apply_table_edits(ds, df, edited, pix, A, off):
    if edited.shape == df.shape and edited.index.equals(df.index) and \
            np.allclose(edited.to_numpy(float), df.to_numpy(float), equal_nan=True):
        return
    if edited[["x", "y"]].isna().any().any():
        st.caption("Enter both x and y for the new row.")
        return
    new_pts = []
    try:
        for idx, row in edited.iterrows():
            if idx in df.index and np.allclose([row.x, row.y], df.loc[idx, ["x", "y"]].to_numpy(float)):
                new_pts.append(list(pix[df.index.get_loc(idx)]))
            else:
                new_pts.append(list(data_to_pixel([[row.x, row.y]], A, off, ss.xlog, ss.ylog)[0]))
    except ValueError as e:
        st.error(str(e))
        return
    snapshot()
    ds["points"] = new_pts
    ss.editor_ver += 1
    st.rerun()


def results_section(A, off, file_stem: str):
    st.divider()
    st.subheader("Results")
    if A is None:
        st.caption("Calibrate the axes and click some points to see the results here.")
        return
    export = []
    for ds in ss.datasets:
        data = pixel_to_data(ds["points"], A, off, ss.xlog, ss.ylog) if ds["points"] else []
        data = [list(r) for r in data]
        if ss.sort_x:
            data.sort(key=lambda r: r[0])
        export.append({"name": ds["name"], "data": data})
    csv_text = build_csv(export, ss.xlabel, ss.ylabel)
    filled = [e for e in export if e["data"]]

    c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="center")
    c1.download_button("⬇️ Download CSV", csv_text.encode("utf-8-sig"),
                       file_name=f"{file_stem or 'digitized'}_data.csv", mime="text/csv",
                       type="primary", disabled=not filled, width="stretch")
    c2.checkbox("Sort each dataset by x", key="sort_x")
    names = [e["name"] for e in filled]
    if len(set(names)) < len(names):
        c3.warning("Two datasets share a name. Rename one so the CSV is clear.")
    if not filled:
        st.caption("No points yet.")
        return
    rows = list(csv.reader(io.StringIO(csv_text)))
    header = [f"{rows[0][i] if i % 2 == 0 else rows[0][i - 1]} · {rows[1][i]}"
              for i in range(len(rows[1]))]
    seen = {}
    for i, h in enumerate(header):  # st.dataframe needs unique column names
        seen[h] = seen.get(h, 0) + 1
        if seen[h] > 1:
            header[i] = f"{h} ({seen[h]})"
    st.dataframe(pd.DataFrame(rows[2:], columns=header), width="stretch", hide_index=True,
                 height=min(38 + 35 * (len(rows) - 2), 420))


def welcome():
    st.markdown(
        """
Extract numbers from a picture of a graph when the raw data isn't available.

1. **Upload** an image of the plot in the sidebar, or try the sample plot.
2. **Calibrate** by clicking two known points on each axis and typing their values.
   Log axes and crooked scans are supported.
3. **Digitize** by clicking the data points. Use several datasets for several curves.
4. **Download** the CSV.
"""
    )
    st.button("Try a sample plot", type="primary", on_click=cb_set, kwargs={"use_sample": True})


def image_panel(img: Image.Image, img_hash: str):
    view = compute_view(img.width, img.height, ss.zoom, ss.pan_x, ss.pan_y)
    disp_w = int(ss.disp_w)
    highlight = ss.calib_next if ss.mode == MODE_CAL else None
    disp = render_view(img, view, disp_w, ss.calib, ss.datasets, ss.active, highlight, ss.connect)
    kwargs = dict(key=f"img_{img_hash}", width=disp.width)
    try:
        value = streamlit_image_coordinates(disp, cursor="crosshair", **kwargs)
    except TypeError:  # older component versions have no cursor option
        value = streamlit_image_coordinates(disp, **kwargs)
    if value and value != ss.last_click:
        ss.last_click = value
        x0, y0, cw, ch = view
        sx = cw / (value.get("width") or disp.width)
        sy = ch / (value.get("height") or disp.height)
        tol = 14 * sx
        handle_click(x0 + value["x"] * sx, y0 + value["y"] * sy, tol)
        st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="NE111 Plot Digitizer", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    data, stem = sidebar_image()
    st.title("📈 Plot Digitizer")
    if data is None:
        welcome()
        return

    img_hash = hashlib.md5(data).hexdigest()[:12]
    if ss.img_hash != img_hash:
        ss.img_hash = img_hash
        reset_work()
    try:
        img, info = cached_decode(data)
    except Exception as e:  # corrupt or unsupported file
        st.error(f"Could not read this image: {e}")
        return
    sidebar_settings(img, info)

    A, off, cal_err = calibration()
    n_pts = sum(len(ds["points"]) for ds in ss.datasets)
    step_tracker(True, A is not None, n_pts)

    top1, top2 = st.columns([4, 1], vertical_alignment="center")
    ss.mode_radio = ss.mode
    top1.radio("Mode", MODES, key="mode_radio", horizontal=True, label_visibility="collapsed",
               on_change=cb_copy, args=("mode_radio", "mode"))
    top2.button("↶ Undo", on_click=undo, disabled=not ss.history, width="stretch",
                help="Undo the last change (points, datasets or calibration)")
    instruction(A is not None, cal_err)

    left, right = st.columns([3, 2], gap="large")
    with left:
        image_panel(img, img_hash)
    with right:
        if ss.mode == MODE_CAL:
            calibration_panel(A is not None, cal_err)
        else:
            datasets_panel(A, off)

    results_section(A, off, stem)


if __name__ == "__main__":
    main()
