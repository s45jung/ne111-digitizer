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

import base64
import tempfile

# --------------------------------------------------------------------------
# Clickable-image component (self-contained, written to a temp folder)
# --------------------------------------------------------------------------
# Shows the plot at the column width, reports clicks as fractions of the image
# size, and draws a magnifier loupe under the cursor for precise placement.

CLICK_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:transparent;overflow:hidden}
#wrap{position:relative;width:100%;line-height:0}
#img{width:100%;height:auto;display:block;cursor:crosshair;user-select:none;
     -webkit-user-drag:none;border-radius:6px}
#loupe{position:absolute;pointer-events:none;width:160px;height:160px;border-radius:50%;
       border:2px solid #fff;box-shadow:0 0 0 1px #000,0 8px 24px rgba(0,0,0,.55);
       background-repeat:no-repeat;background-color:#fff;display:none}
#loupe:before,#loupe:after{content:"";position:absolute;background:rgba(255,40,40,.85)}
#loupe:before{left:50%;top:22%;bottom:22%;width:1px}
#loupe:after{top:50%;left:22%;right:22%;height:1px}
</style></head><body><div id="wrap"><img id="img" alt="plot"><div id="loupe"></div></div>
<script>
const img=document.getElementById("img"), loupe=document.getElementById("loupe");
let mag=3, lastH=-1;
function send(type,data){window.parent.postMessage(Object.assign({isStreamlitMessage:true,type:type},data),"*");}
function setH(){const h=Math.ceil(document.getElementById("wrap").getBoundingClientRect().height);
  if(h!==lastH){lastH=h;send("streamlit:setFrameHeight",{height:h});}}
window.addEventListener("message",function(e){
  if(!e.data||e.data.type!=="streamlit:render")return;
  const a=e.data.args; mag=a.magnify||0;
  if(img.getAttribute("src")!==a.src){img.setAttribute("src",a.src);}
  loupe.style.backgroundImage="url('"+a.src+"')"; setH();});
img.onload=setH; new ResizeObserver(setH).observe(document.body);
img.addEventListener("click",function(ev){const r=img.getBoundingClientRect();
  send("streamlit:setComponentValue",{dataType:"json",value:{
    x:(ev.clientX-r.left)/r.width,y:(ev.clientY-r.top)/r.height,t:Date.now()}});});
img.addEventListener("mousemove",function(ev){
  if(mag<=1){loupe.style.display="none";return;}
  const r=img.getBoundingClientRect(), x=ev.clientX-r.left, y=ev.clientY-r.top, R=80;
  loupe.style.display="block"; loupe.style.left=(x-R)+"px"; loupe.style.top=(y-R)+"px";
  loupe.style.backgroundSize=(r.width*mag)+"px "+(r.height*mag)+"px";
  loupe.style.backgroundPosition=(R-x*mag)+"px "+(R-y*mag)+"px";});
img.addEventListener("mouseleave",function(){loupe.style.display="none";});
send("streamlit:componentReady",{apiVersion:1});
</script></body></html>"""

try:
    import streamlit.components.v1 as _components
    _comp_dir = Path(tempfile.gettempdir()) / "ne111_plot_click_v1"
    _comp_dir.mkdir(exist_ok=True)
    (_comp_dir / "index.html").write_text(CLICK_HTML, encoding="utf-8")
    _plot_click = _components.declare_component("plot_click", path=str(_comp_dir))
except Exception:  # lets the pure functions below be imported for testing
    _plot_click = None


def clickable_image(img: Image.Image, key: str, magnify: float):
    """Show `img`; return {"x": fx, "y": fy, "t": ms} (fractions of size) on click."""
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=False, compress_level=3)
    src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return _plot_click(src=src, magnify=magnify, key=key, default=None)

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
MODE_CAL, MODE_ADD, MODE_DEL = "Calibrate", "Add points", "Delete points"
MODES = [MODE_CAL, MODE_ADD, MODE_DEL]
IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp", "webp"]
# Widget values that must survive while their widget is not on screen.
PERSIST_KEYS = ["xlabel", "ylabel", "xlog", "ylog", "X1_val", "X2_val", "Y1_val",
                "Y2_val", "zoom", "pan_x", "pan_y", "sort_x", "connect", "magnify"]
RENDER_W = 1200  # pixel width the plot is drawn at before the browser scales it
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
    font = _font(18)

    # Calibration axes and markers
    for p, q, col in (("X1", "X2", CAL_COLORS["X1"]), ("Y1", "Y2", CAL_COLORS["Y1"])):
        if calib.get(p) and calib.get(q):
            d.line([to_d(calib[p]), to_d(calib[q])], fill=col + "99", width=2)
    for k in CAL_KEYS:
        if calib.get(k) is None:
            continue
        cx, cy = to_d(calib[k])
        col = CAL_COLORS[k]
        r = 14 if k == highlight else 10
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
        r = 7 if i == active else 5.5
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
                "zoom": 1.0, "pan_x": 0.5, "pan_y": 0.5, "magnify": 3.0,
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
# Visual language: pure black, dark-grey cards, white uppercase type with wide
# tracking, small grey section labels, heavy upright headline, underlined tabs.

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400..900&display=swap');
:root { --card:#1c1c1e; --card-hi:#2a2a2c; --line:#2c2c2e; --muted:#8e8e93; --soft:#c7c7cc; }
.stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp td, .stApp th, .dg-root { font-family:'Montserrat', sans-serif; }
header[data-testid="stHeader"] { background:#000; }
.block-container { padding-top:2.4rem; padding-bottom:3rem; max-width:1500px; }
section[data-testid="stSidebar"] { border-right:1px solid var(--line); }

/* top bar, like the phone's "CREATE" header */
.dg-topbar { text-align:center; font-weight:600; letter-spacing:.22em; font-size:.8rem;
             padding:.2rem 0 1.1rem; border-bottom:1px solid var(--line); margin-bottom:1.2rem; }
.dg-eyebrow { text-align:center; letter-spacing:.25em; font-size:.85rem; font-weight:500;
              color:var(--soft); margin-top:2.5rem; }
.dg-hero { text-align:center; font-weight:800; text-transform:uppercase;
           font-size:clamp(1.8rem, 4vw, 3rem); line-height:1.08; letter-spacing:.01em;
           margin:.6rem auto 2.2rem; max-width:900px; }
.dg-label { color:var(--soft); text-transform:uppercase; letter-spacing:.09em; font-size:.74rem;
            font-weight:600; margin:.4rem 0 .55rem; }
.dg-label .r { float:right; color:var(--muted); }

/* list cards */
.dg-card { background:var(--card); border-radius:6px; padding:.95rem 1.1rem; display:flex;
           align-items:center; gap:.8rem; border-left:3px solid transparent; }
.dg-card.active { border-left-color:#fff; background:var(--card-hi); }
.dg-card .t { font-weight:700; text-transform:uppercase; letter-spacing:.06em; font-size:.88rem; }
.dg-card .s { color:var(--muted); text-transform:uppercase; letter-spacing:.07em; font-size:.68rem;
              margin-top:.2rem; }
.dg-card .grow { flex:1; min-width:0; }
.dg-card .dots { color:var(--muted); letter-spacing:.1em; font-weight:700; }
.dg-card .num { font-weight:800; font-size:1.4rem; color:var(--muted); width:2.2rem; }
.swatch { display:inline-block; width:.85rem; height:.85rem; border-radius:50%; flex:none;
          box-shadow:0 0 0 2px #000, 0 0 0 3px #555; }

/* whole dataset card clickable: invisible button stretched over it */
[class*="st-key-dscard_"] { position:relative; gap:0 !important; }
[class*="st-key-dscard_"] [data-testid="stElementContainer"]:has(button) {
    position:absolute; inset:0; z-index:2; margin:0; }
[class*="st-key-dscard_"] [data-testid="stElementContainer"]:has(button) * { width:100%; height:100%; }
[class*="st-key-dscard_"] button { opacity:0; cursor:pointer; }

/* instruction banner */
.dg-banner { background:var(--card); border-radius:6px; padding:.85rem 1.1rem; margin:.2rem 0 1rem;
             border-left:3px solid #fff; }
.dg-banner.warn { border-left-color:#ff9f0a; }
.dg-banner.ok { border-left-color:#30d158; }
.dg-banner .msg { font-size:.95rem; }
.dg-banner .dg-label { margin:0 0 .25rem; }

/* step tracker */
.dg-steps { display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:1rem; }
.dg-step { padding:.35rem .8rem; border-radius:999px; border:1px solid var(--line); color:var(--muted);
           text-transform:uppercase; letter-spacing:.08em; font-size:.68rem; font-weight:600; }
.dg-step.done { color:#fff; }
.dg-step.now { background:#fff; color:#000; border-color:#fff; }

/* mode radio rendered as underlined tabs (WORKOUTS | PLANS) */
.st-key-mode_radio div[role="radiogroup"] { display:flex; gap:0; width:100%; border-bottom:1px solid var(--line); }
.st-key-mode_radio, .st-key-mode_radio [data-testid="stRadio"],
.st-key-mode_radio [data-testid="stRadio"] > div { width:100% !important; }
.st-key-mode_radio div[role="radiogroup"] > * { flex:1; margin:0 !important; }
.st-key-mode_radio [data-testid="stRadioOption"] p { white-space:nowrap; }
.st-key-mode_radio [data-testid="stRadioOption"] { width:100%; justify-content:center; margin:0;
    padding:.65rem 0 .7rem; border-bottom:2px solid transparent; cursor:pointer; }
.st-key-mode_radio [data-testid="stRadioOption"] > div > div:first-child { display:none; }
.st-key-mode_radio [data-testid="stRadioOption"] p { text-transform:uppercase; letter-spacing:.1em;
    font-weight:600; font-size:.82rem; color:var(--muted); }
.st-key-mode_radio [data-testid="stRadioOption"][data-selected="true"] { border-bottom-color:#fff; }
.st-key-mode_radio [data-testid="stRadioOption"][data-selected="true"] p { color:#fff; }
.st-key-mode_radio [data-testid="stRadioOption"]:hover p { color:#fff; }

/* buttons */
.stButton button, .stDownloadButton button, [data-testid="stPopover"] button {
    background:var(--card); border:1px solid transparent; color:#fff; border-radius:6px; }
.stButton button p, .stDownloadButton button p, [data-testid="stPopover"] button p {
    text-transform:uppercase; letter-spacing:.08em; font-weight:600; font-size:.74rem; }
.stButton button:hover, .stDownloadButton button:hover, [data-testid="stPopover"] button:hover {
    background:var(--card-hi); border-color:#3a3a3c; color:#fff; }
button[data-testid="stBaseButton-primary"] { background:#fff !important; color:#000 !important; }
button[data-testid="stBaseButton-primary"] p { color:#000 !important; }
button[data-testid="stBaseButton-primary"]:hover { background:#d9d9d9 !important; }
.stButton button:disabled { opacity:.35; }

/* inputs and cards */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    background:var(--card) !important; border-color:transparent !important; }
[data-testid="stFileUploaderDropzone"] { background:var(--card); border:1px dashed #3a3a3c; }
[class*="st-key-card_"] { background:var(--card); border:none !important; }
.stApp label p { text-transform:uppercase; letter-spacing:.07em; font-size:.7rem; font-weight:600;
                 color:var(--soft); }
.st-key-mode_radio label p { font-size:.82rem; }
[data-testid="stCaptionContainer"] { color:var(--muted); }
hr { border-color:var(--line) !important; }
</style>
"""


def md_to_html(text: str) -> str:
    """Tiny **bold** -> <b>bold</b> converter for banner messages."""
    parts = text.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def label(text: str, right: str = "", where=st):
    r = f"<span class='r'>{right}</span>" if right else ""
    where.markdown(f"<div class='dg-label'>{text}{r}</div>", unsafe_allow_html=True)


def banner(step: str, msg: str, kind: str = ""):
    st.markdown(f"<div class='dg-banner {kind}'><div class='dg-label'>{step}</div>"
                f"<div class='msg'>{md_to_html(msg)}</div></div>", unsafe_allow_html=True)


def sidebar_image() -> tuple[bytes | None, str]:
    label("Image", where=st.sidebar)
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


@st.cache_data(show_spinner=False)
def cached_sample() -> bytes:
    return make_sample_plot()


@st.cache_data(show_spinner=False, max_entries=4)
def cached_decode(data: bytes):
    return decode_image(data)


def sidebar_settings(img: Image.Image, info: dict):
    kind = "greyscale" if info["greyscale"] else "colour"
    frames = f" · frame 1 of {info['frames']}" if info["frames"] > 1 else ""
    st.sidebar.caption(f"{info['format']} · {info['size'][0]}×{info['size'][1]} px · {kind}{frames}")

    st.sidebar.write("")
    label("Axis labels", where=st.sidebar)
    st.sidebar.text_input("x-axis label", key="xlabel", help="Used as the column header in the CSV")
    st.sidebar.text_input("y-axis label", key="ylabel")

    st.sidebar.write("")
    label("View", where=st.sidebar)
    st.sidebar.select_slider("Magnifier", options=[1.0, 2.0, 3.0, 4.0, 6.0], key="magnify",
                             format_func=lambda m: "Off" if m == 1 else f"{m:g}×",
                             help="Loupe that follows the cursor over the plot")
    st.sidebar.select_slider("Zoom", options=[1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], key="zoom",
                             format_func=lambda z: f"{z:g}×",
                             help="Zoom in to place points more precisely")
    if ss.zoom > 1:
        st.sidebar.slider("Pan left / right", 0.0, 1.0, key="pan_x", step=0.01)
        st.sidebar.slider("Pan up / down", 0.0, 1.0, key="pan_y", step=0.01)
        view = compute_view(img.width, img.height, ss.zoom, ss.pan_x, ss.pan_y)
        thumb = img.copy()
        thumb.thumbnail((300, 300))
        k = thumb.width / img.width
        x0, y0, cw, ch = view
        ImageDraw.Draw(thumb).rectangle(
            [x0 * k, y0 * k, (x0 + cw) * k, (y0 + ch) * k], outline="#ff2b2b", width=3)
        st.sidebar.image(thumb, caption="Visible region")
    st.sidebar.checkbox("Connect points with lines", key="connect")


def step_tracker(calibrated: bool, n_pts: int):
    steps = [("Upload", True), ("Calibrate", calibrated), ("Digitize", n_pts > 0), ("Export", False)]
    out, current_found = [], False
    for i, (name, done) in enumerate(steps, 1):
        cls = "done" if done else ("now" if not current_found else "")
        current_found = current_found or not done
        out.append(f"<span class='dg-step {cls}'>{'✓' if done else f'0{i}'}&nbsp; {name}</span>")
    st.markdown(f"<div class='dg-steps'>{''.join(out)}</div>", unsafe_allow_html=True)


def instruction(calibrated: bool, cal_err: str | None):
    if ss.mode == MODE_CAL:
        if ss.calib_next:
            banner("Step 2 · Calibrate",
                   f"Click the image to place **{ss.calib_next}**, {CAL_HELP[ss.calib_next]}.")
        elif calibrated:
            banner("Step 2 · Calibrate",
                   "Calibration done. Check the values on the right, then start digitizing.", "ok")
        else:
            banner("Step 2 · Calibrate", cal_err or "", "warn")
    elif ss.mode == MODE_ADD:
        if not calibrated:
            banner("Step 3 · Digitize", f"Calibrate the axes first. {cal_err}", "warn")
        else:
            ds = ss.datasets[ss.active]
            banner("Step 3 · Digitize",
                   f"Click data points to add them to **{ds['name']}**. "
                   "Use **+ New** for another curve.")
    else:
        banner("Edit", "Click a point on the image to delete it.")


def calibration_panel(calibrated: bool, cal_err: str | None):
    label("Axis calibration")
    st.caption("Place two points on each axis at values you know (labelled ticks are best). "
               "Rotated or skewed scans are handled automatically.")
    for axis, keys in (("x", ("X1", "X2")), ("y", ("Y1", "Y2"))):
        with st.container(border=True, key=f"card_{axis}"):
            st.toggle(f"Logarithmic {axis}-axis", key=f"{axis}log")
            for k in keys:
                c1, c2 = st.columns([1.1, 1])
                placed = ss.calib[k] is not None
                is_next = ss.calib_next == k
                text = f"{'✓ Move' if placed else 'Place'} {k}"
                c1.button(text, key=f"pick_{k}", type="primary" if is_next else "secondary",
                          on_click=cb_set, kwargs={"calib_next": k}, width="stretch",
                          help=f"Then click {CAL_HELP[k]}")
                c2.number_input(f"{k} value", key=f"{k}_val", format="%g",
                                label_visibility="collapsed")
    tx, ty = axis_tilt(ss.calib)
    if calibrated:
        msg = f"x-axis tilt {tx:+.2f}° · y-axis tilt {ty:+.2f}°"
        if max(abs(tx), abs(ty)) > 0.3:
            msg += " · rotation corrected"
        st.caption(msg)
        st.button("Start digitizing →", type="primary", width="stretch",
                  on_click=cb_set, kwargs={"mode": MODE_ADD})
    elif cal_err and not ss.calib_next:
        st.error(cal_err)
    st.button("Reset calibration", on_click=cb_reset_calibration, width="stretch")


def dataset_cards():
    """List of datasets styled like the reference's workout cards; click to select."""
    for i, ds in enumerate(ss.datasets):
        n = len(ds["points"])
        active = i == ss.active
        with st.container(key=f"dscard_{ds['uid']}"):
            st.markdown(
                f"<div class='dg-card {'active' if active else ''}'>"
                f"<span class='swatch' style='background:{ds['color']}'></span>"
                f"<div class='grow'><div class='t'>{ds['name']}</div>"
                f"<div class='s'>{n} point{'s' if n != 1 else ''}"
                f"{' · active' if active else ''}</div></div>"
                f"<span class='dots'>•••</span></div>", unsafe_allow_html=True)
            st.button(f"Select {ds['name']}", key=f"sel_{ds['uid']}",
                      on_click=cb_set, kwargs={"active": i})


def datasets_panel(A, off):
    h1, h2 = st.columns([3, 1.2], vertical_alignment="bottom")
    with h1:
        label("Datasets", f"{len(ss.datasets)} total")
    h2.button("+ New", on_click=cb_add_dataset, width="stretch", help="Add another dataset")
    ss.active = min(ss.active, len(ss.datasets) - 1)
    dataset_cards()

    ds = ss.datasets[ss.active]
    st.write("")
    label("Edit selected")
    c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
    ds["name"] = c1.text_input("Name", value=ds["name"], key=f"name_{ds['uid']}").strip() \
        or f"Dataset {ss.active + 1}"
    ds["color"] = c2.color_picker("Colour", value=ds["color"], key=f"color_{ds['uid']}")

    b1, b2, b3 = st.columns(3)
    b1.button("↶ Last pt", on_click=cb_remove_last, args=(ss.active,), width="stretch",
              disabled=not ds["points"], help="Remove the last point of this dataset")
    b2.button("Clear", on_click=cb_clear, args=(ss.active,), width="stretch",
              disabled=not ds["points"], help="Remove all points of this dataset")
    with b3.popover("Delete", width="stretch"):
        st.write(f"Delete **{ds['name']}** and its {len(ds['points'])} points?")
        st.button("Yes, delete", type="primary", on_click=cb_delete_dataset, args=(ss.active,))

    st.write("")
    label("Points", "edit to move marks")
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
    if A is None:
        label("Results")
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
    total = sum(len(e["data"]) for e in filled)
    label("Results", f"{len(filled)} datasets · {total} points")

    c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="center")
    c1.download_button("Download CSV", csv_text.encode("utf-8-sig"),
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
    st.markdown("<div class='dg-eyebrow'>PLOT DIGITIZER</div>"
                "<div class='dg-hero'>Turn any graph into<br>data you can use</div>",
                unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        label("How it works")
        steps = [("Upload", "PNG, JPG, GIF or TIFF · colour or greyscale"),
                 ("Calibrate", "Two known points per axis · linear or log · crooked scans OK"),
                 ("Digitize", "Click the data · one colour per dataset"),
                 ("Export", "CSV with dataset names and axis labels")]
        for i, (t, s) in enumerate(steps, 1):
            st.markdown(f"<div class='dg-card' style='margin-bottom:.5rem'>"
                        f"<span class='num'>0{i}</span><div class='grow'><div class='t'>{t}</div>"
                        f"<div class='s'>{s}</div></div></div>", unsafe_allow_html=True)
        st.write("")
        st.button("Try a sample plot", type="primary", width="stretch",
                  on_click=cb_set, kwargs={"use_sample": True})
        st.caption("Or upload your own image in the sidebar.")


def image_panel(img: Image.Image, img_hash: str):
    view = compute_view(img.width, img.height, ss.zoom, ss.pan_x, ss.pan_y)
    highlight = ss.calib_next if ss.mode == MODE_CAL else None
    disp = render_view(img, view, RENDER_W, ss.calib, ss.datasets, ss.active, highlight, ss.connect)
    # Rendered sharp at RENDER_W, then scaled by the browser to fit the column.
    value = clickable_image(disp, key=f"img_{img_hash}", magnify=ss.magnify)
    if value and value != ss.last_click:
        ss.last_click = value
        x0, y0, cw, ch = view
        # tolerance for delete: ~1.5% of the visible width
        handle_click(x0 + value["x"] * cw, y0 + value["y"] * ch, 0.015 * cw)
        st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    data, stem = sidebar_image()
    if data is None:
        welcome()
        return
    st.markdown("<div class='dg-topbar'>PLOT DIGITIZER</div>", unsafe_allow_html=True)

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

    top1, top2 = st.columns([5, 1], vertical_alignment="bottom")
    with top1:
        step_tracker(A is not None, n_pts)
    top2.button("↶ Undo", on_click=undo, disabled=not ss.history, width="stretch",
                help="Undo the last change (points, datasets or calibration)")
    ss.mode_radio = ss.mode
    st.radio("Mode", MODES, key="mode_radio", horizontal=True, label_visibility="collapsed",
             on_change=cb_copy, args=("mode_radio", "mode"))
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
