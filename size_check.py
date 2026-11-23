"""Pill size check in pixels: length, width and the two hole openings, measured on the template-matched crop.

    python size_check.py data/raw/defect/*.bmp                        # SKU sku01, nominal from its good photos
    python size_check.py some.bmp --sku sku01 --good data/raw/good --out outputs/size

To use it in another app (sku = main.load_sku(...): template matcher, spec_mm, hole_x, hole_radius):

    import size_check
    calibration = size_check.calibrate([size_check.measure(cv2.imread(p), sku) for p in good_paths], sku['spec_mm'])
    result = size_check.judge(size_check.measure(cv2.imread('raw.bmp'), sku), calibration)
    result['bad'], result['length'], result['width'], result['hole1'], result['hole2']

Nominal = median of the good pills, tolerance = datasheet tolerance in % of nominal.
mm = px x mm_per_px, one scale: datasheet length / median length of the good pills.
Every step drawn on one image: size_check.ipynb.

- Crop: the SKU's template matcher (configs/<sku>.json), PAD px around the template size.
- Length: rows near the pill axis, walking out to each end. The rounded end only fades in brightness, so the
  first edge is where the colour distance from the pill (Lab a,b) crosses halfway between the pill and the
  background level; then each row moves out to the rim, a thin light/dark line on the outline, because where
  the background has the pill's colour (defect/12) the colour edge alone stops ~30 px inside.
- Width: columns between the ends, walking out from inside the pill; the first strong brightness step is the
  pill edge (the clamp ring further out is stronger).
- Hole: outer edge of the countersink, the HALCON 2D-metrology way (written from the reference manual):
  48 measure regions across the expected circle, Gaussian derivative, last dark->light edge going outward,
  RANSAC circle. Circle, not ellipse: this edge stays round, and an ellipse stretches when an arc is missing.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

PAD = 120                    # margin around the crop so the edges are inside
INNER = 200                  # length: caliper start, px inside each crop end (before the hole openings)
ROWS = 40                    # length: rows within +-ROWS of the pill axis
RIM_IN, RIM_OUT = 10, 50     # length: look for the rim from RIM_IN px inside to RIM_OUT px outside the colour edge
EDGE_IN, EDGE_OUT = 35, 25   # width: caliper from EDGE_IN inside to EDGE_OUT outside the crop border
GAP = 3                      # step = brightness difference over GAP px (1 px differences are mostly noise)
RATIO = 0.5                  # edge = first step / outermost rim peak stronger than RATIO x the strongest
RATIO_BOT = 0.3              # bottom edge: the pill edge is weaker than the ring edge below it

NUM_MEASURES = 48            # hole: measure regions around the circle
MEASURE_LENGTH1 = 45         # half length across the circle (px); covers the hole sitting ~35 px off the expected spot
MEASURE_LENGTH2 = 5          # half width along the circle (px), averaged into the profile
MEASURE_SIGMA = 2.0          # edges are soft (out-of-focus rounded rim), HALCON default is 1.0
MEASURE_THRESHOLD = 2.0      # min derivative in grey levels per px
DISTANCE_THRESHOLD = 3.5     # RANSAC inlier distance (px), HALCON default
MIN_SCORE = 0.6              # inliers / measure regions (HALCON default 0.7); tilted pills lose the dim top arc (defect/14)
NUM_ITERATIONS = 300
RAND_SEED = 42

A, T = PAD + INNER, PAD + EDGE_IN  # left length caliper start column, top width caliper start row
DIMS = {'length': 'length', 'width': 'width', 'hole1': 'hole', 'hole2': 'hole'}  # result key -> spec_mm key


def geometry(sku):
    """Positions in the crop that depend on the SKU's template size (TH x TW) and hole place."""
    th, tw = sku['matcher'].height, sku['matcher'].width
    mid = PAD + th // 2                       # pill axis row in the crop
    return dict(TH=th, TW=tw, MID=mid,
                B=PAD + tw - INNER,           # right length caliper start column (inside the pill, before the holes)
                U=PAD + th - EDGE_IN,         # bottom width caliper start row
                HOLE_CENTRES=((PAD + sku['hole_x'], mid), (PAD + tw - sku['hole_x'], mid)))  # expected


# ---------- edges ----------

def smooth_gray(crop):
    return cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)  # blur in float: no rounding


def colour_dist(crop):
    """Per pixel: distance in Lab a,b from the pill colour (median just inside the left caliper start)."""
    lab = cv2.GaussianBlur(cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_BGR2Lab), (0, 0), 2)
    mid = crop.shape[0] // 2  # = geometry()['MID']: the crop is the template plus PAD on both sides
    ref = np.median(lab[mid - ROWS:mid + ROWS, A:A + 60, 1:].reshape(-1, 2), 0)
    return np.linalg.norm(lab[..., 1:] - ref, axis=2)


def half_level(prof, n=50):
    """prof: (samples, lines), scan starts inside the region. Per line: first sample past the level halfway
    between the inside level (first n samples) and the outside level (last n samples)."""
    inside, outside = np.median(prof[:n], 0), np.median(prof[-n:], 0)
    return ((prof - (inside + outside) / 2) * np.sign(outside - inside) > 0).argmax(0)


def rim(grad, ys, edges, step):
    """Per row: move the colour edge to the outermost peak of grad (>= RATIO x the strongest in the window)
    between RIM_IN inside and RIM_OUT outside it. step -1 = walking left, +1 = walking right."""
    out = edges.copy()
    for i, (y, e) in enumerate(zip(ys, edges)):
        xs = np.clip(e + step * np.arange(-RIM_IN, RIM_OUT), 0, grad.shape[1] - 1)
        v = grad[y, xs]
        k = np.flatnonzero((v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] >= RATIO * v.max())) + 1
        if len(k):
            out[i] = xs[k[-1]]
    return out


def caliper(prof, ratio=RATIO):
    """prof: (samples along the scan, lines), scan starts on the clean side.
    Per line: index of the first step stronger than ratio x the line's strongest step, moved to the step's peak."""
    d = np.abs(prof[GAP:] - prof[:-GAP])
    i = (d > ratio * d.max(0)).argmax(0)
    cols = np.arange(d.shape[1])
    return i + np.stack([d[np.minimum(i + k, len(d) - 1), cols] for k in range(8)]).argmax(0) + GAP // 2


# ---------- hole ----------

def fit_circle(points):
    """Least-squares circle (Kasa: x^2 + y^2 + D x + E y + F = 0). Returns (cx, cy, r) or None."""
    x, y = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    (D, E, F), *_ = np.linalg.lstsq(np.stack([x, y, np.ones_like(x)], 1), -(x ** 2 + y ** 2), rcond=None)
    r2 = D ** 2 / 4 + E ** 2 / 4 - F
    return (-D / 2, -E / 2, np.sqrt(r2)) if r2 > 0 else None


def measure_edges(gray, center, radius):
    """One edge point per measure region (NaN where none passes the threshold).
    Last dark->light edge going outward: the inner bore edge has the same polarity and lies further in."""
    phi = 2 * np.pi * np.arange(NUM_MEASURES) / NUM_MEASURES
    normal = np.stack([np.cos(phi), np.sin(phi)], 1)          # inside -> outside
    tangent = np.stack([-np.sin(phi), np.cos(phi)], 1)
    s = np.arange(-MEASURE_LENGTH1, MEASURE_LENGTH1 + 1, dtype=np.float32)
    t = np.arange(-MEASURE_LENGTH2, MEASURE_LENGTH2 + 1, dtype=np.float32)
    base = np.asarray(center, np.float32) + radius * normal    # region centres on the expected circle
    pts = (base[:, None, None] + s[None, :, None, None] * normal[:, None, None]
           + t[None, None, :, None] * tangent[:, None, None]).astype(np.float32)   # remap wants float32 maps
    samples = cv2.remap(gray, pts[..., 0].reshape(-1, len(t)), pts[..., 1].reshape(-1, len(t)), cv2.INTER_LINEAR)
    profiles = samples.reshape(NUM_MEASURES, len(s), len(t)).mean(2)   # slices across the region
    d = gaussian_filter1d(profiles, MEASURE_SIGMA, axis=1, order=1)
    peak = (d[:, 1:-1] > d[:, :-2]) & (d[:, 1:-1] >= d[:, 2:]) & (d[:, 1:-1] > MEASURE_THRESHOLD)
    points = np.full((NUM_MEASURES, 2), np.nan, np.float32)
    for i in range(NUM_MEASURES):
        k = np.flatnonzero(peak[i]) + 1
        if len(k) == 0:
            continue
        k = k[-1]
        a, b, c = d[i, k - 1], d[i, k], d[i, k + 1]
        sub = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c != 0 else 0.0   # parabola through the peak
        points[i] = base[i] + (s[k] + sub) * normal[i]
    return points


def find_hole(gray, center, radius):
    """gray: float32 grayscale crop, center: expected (x, y), radius: expected radius (px).
    Returns dict(center, radius, points, used) or None when too few edges fit one circle."""
    points = measure_edges(gray, center, radius)
    found = ~np.isnan(points[:, 0])
    pts = points[found]
    if len(pts) < 3:
        return None
    rng = np.random.default_rng(RAND_SEED)
    best = np.zeros(len(pts), bool)
    for _ in range(NUM_ITERATIONS):
        c = fit_circle(pts[rng.choice(len(pts), 3, replace=False)])   # 3 points = exact circle
        if c is None or not np.isfinite(c[2]):
            continue
        inliers = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - c[2]) < DISTANCE_THRESHOLD
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(3, MIN_SCORE * NUM_MEASURES):
        return None
    cx, cy, r = fit_circle(pts[best])
    used = np.zeros(NUM_MEASURES, bool)
    used[np.flatnonzero(found)[best]] = True
    return dict(center=(cx, cy), radius=r, points=points, used=used)


# ---------- measure / judge / draw ----------

def measure(image, sku):
    """Raw BGR image -> dict of the three measurements (px), plus the per-line edges and the crop for drawing."""
    g = geometry(sku)
    TH, MID, B, U = g['TH'], g['MID'], g['B'], g['U']
    crop, center, angle, _ = sku['matcher'].crop(image, pad=PAD)
    rows = colour_dist(crop)[MID - ROWS:MID + ROWS]
    lefts0 = A - half_level(rows[:, A::-1].T)        # colour edge per row, walking left
    rights0 = B + half_level(rows[:, B:].T)          # walking right
    grad = np.abs(cv2.Sobel(cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 1.5), cv2.CV_32F, 1, 0, ksize=3)).sum(2)
    ys = np.arange(MID - ROWS, MID + ROWS)
    lefts, rights = rim(grad, ys, lefts0, -1), rim(grad, ys, rights0, 1)
    cols = smooth_gray(crop)[:, A:B]
    tops = T - caliper(cols[T:PAD - EDGE_OUT:-1])    # one edge per column, walking up
    bots = U + caliper(cols[U:PAD + TH + EDGE_OUT], RATIO_BOT)  # walking down
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)   # unblurred: the hole derivative smooths itself
    holes = [find_hole(gray, c, sku['hole_radius']) for c in g['HOLE_CENTRES']]
    hole_d = [2 * h['radius'] if h else np.nan for h in holes]        # NaN fails the check
    left, right, top, bot = np.median(lefts), np.median(rights), np.median(tops), np.median(bots)
    return dict(left=left, right=right, top=top, bot=bot, length=right - left, width=bot - top,
                hole1=hole_d[0], hole2=hole_d[1], holes=holes, hole_centres=g['HOLE_CENTRES'],
                lefts=lefts, rights=rights, lefts0=lefts0, rights0=rights0, tops=tops, bots=bots,
                crop=crop, center=center, angle=angle, M=sku['matcher'].crop_matrix(center, angle, PAD))


def hole_mid(r):
    """x of the midpoint of the two holes (expected spot for a hole not found)."""
    return np.mean([h['center'][0] if h else c[0] for h, c in zip(r['holes'], r['hole_centres'])])


def calibrate(goods, spec):
    """From measure() results of good pills and the datasheet spec ({dim: (nominal, +-tol)} in mm): nominal
    (median), tolerance (px) and the good ends' distance from the hole midpoint (DL, DR)."""
    nominal = {d: float(np.median([r[d] for r in goods])) for d in ('length', 'width')}
    nominal['hole1'] = nominal['hole2'] = float(np.nanmedian([r[h] for r in goods for h in ('hole1', 'hole2')]))  # one hole not found must not fail every pill
    limit = {d: nominal[d] * spec[s][1] / spec[s][0] for d, s in DIMS.items()}
    return dict(nominal=nominal, limit=limit, mm_per_px=spec['length'][0] / nominal['length'],
                DL=float(np.median([hole_mid(r) - r['left'] for r in goods])),
                DR=float(np.median([r['right'] - hole_mid(r) for r in goods])))


def judge(r, cal):
    """Adds the standard ends L0/R0 and the failing dims ('bad') to a measure() result.
    The standard is anchored on the hole midpoint: holes do not move when an end is cut, the template match does."""
    r['L0'], r['R0'] = hole_mid(r) - cal['DL'], hole_mid(r) + cal['DR']
    r['bad'] = [d for d in DIMS if not abs(r[d] - cal['nominal'][d]) <= cal['limit'][d]]   # 'not <=' so NaN fails
    return r


def draw_overlay(image, r, cal, half=900):
    """Raw image around the pill: standard box (yellow), measured box and holes (green ok / red fail),
    and a red line with the gap in mm from each short end to the standard end."""
    red, green, yellow = (0, 0, 255), (0, 255, 0), (0, 255, 255)
    inverse = cv2.invertAffineTransform(r['M'])
    to_raw = lambda pts: cv2.transform(np.float32(pts).reshape(-1, 1, 2), inverse).reshape(-1, 2).astype(np.int32)
    box = lambda x0, x1, y0, y1: [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    img = image.copy()
    mid, w0 = (r['top'] + r['bot']) / 2, cal['nominal']['width'] / 2
    cv2.polylines(img, [to_raw(box(r['L0'], r['R0'], mid - w0, mid + w0))], True, yellow, 6)
    cv2.polylines(img, [to_raw(box(r['left'], r['right'], r['top'], r['bot']))], True,
                  red if {'length', 'width'} & set(r['bad']) else green, 6)
    for h, d in zip(r['holes'], ('hole1', 'hole2')):
        if h:
            cv2.circle(img, tuple(int(v) for v in to_raw([h['center']])[0]), int(round(h['radius'])), red if d in r['bad'] else green, 6)
    if 'length' in r['bad']:
        for end, std in ((r['left'], r['L0']), (r['right'], r['R0'])):
            if abs(end - std) > cal['limit']['length'] / 2:
                p, q = to_raw([(end, mid), (std, mid)])
                cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), red, 10)
                cv2.putText(img, f"{-abs(end - std) * cal['mm_per_px']:.2f}", tuple(int(v) for v in p + (-150, -60)), cv2.FONT_HERSHEY_SIMPLEX, 4, red, 10)
    c = to_raw([(r['crop'].shape[1] / 2, mid)])[0]  # crop centre = PAD + TW / 2
    return img[max(c[1] - half, 0):c[1] + half, max(c[0] - half, 0):c[0] + half]


def image_paths(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.bmp', '.png', '.jpg'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', nargs='+')
    parser.add_argument('--sku', default='sku01', help='SKU config in configs/ (template, spec, hole place)')
    parser.add_argument('--good', help="folder of good raw images (gives the nominal), default: the SKU's good")
    parser.add_argument('--out', default='outputs/size', help='folder for overlay images')
    args = parser.parse_args()

    from main import load_sku  # here, not at the top: main imports this module
    sku = load_sku(args.sku)
    good = args.good or sku['good']
    good_paths = image_paths(good)
    assert len(good_paths) >= 2, f'need at least 2 good images in {good}'
    goods = []
    for path in good_paths:
        image = cv2.imread(str(path))
        assert image is not None, path
        goods.append(measure(image, sku))
    cal = calibrate(goods, sku['spec_mm'])   # shortcut: re-measured from --good on every run; save it to a file if startup time matters
    for d in ('length', 'width', 'hole1'):
        print(f"{d:6} nominal {cal['nominal'][d]:6.1f} px +- {cal['limit'][d]:4.1f} px")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in args.images:
        image = cv2.imread(image_path)
        if image is None:
            print(f'{image_path}: cannot read, skipped')
            continue
        r = judge(measure(image, sku), cal)
        print(f"{image_path}: {'FAIL ' + ','.join(r['bad']) if r['bad'] else 'ok'}  length {r['length']:.0f}  "
              f"width {r['width']:.0f}  holes {r['hole1']:.0f} {r['hole2']:.0f}  "
              f"ends short by {r['left'] - r['L0']:+.0f} / {r['R0'] - r['right']:+.0f} px")
        out_name = f'{Path(image_path).parent.name}_{Path(image_path).stem}.png'
        cv2.imwrite(str(out_dir / out_name), draw_overlay(image, r, cal))


if __name__ == '__main__':
    main()
