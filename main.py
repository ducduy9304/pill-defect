"""SKU configs and the two pill checks (surface, dimensions) on a whole camera photo (BGR).
Used by the app (app/demo.py), train.py, infer.py and size_check.py. It also runs those scripts by name:

    python main.py train [args]                        # = python train.py [args]
    python main.py infer data/raw/defect/*.bmp         # = python infer.py ...
    python main.py size [args]                         # = python size_check.py [args]

    import main
    sku = main.load_sku('sku01')
    prepared = main.prepare(sku, 0)                    # 0 = surface, 1 = dimensions (main.STEPS)
    result = main.check(photo, prepared)
    result['verdict'], result['faults'], result['summary'], result['image'], result['process']

One file per SKU, configs/<sku>.json (paths relative to the repo):
    template          the pill cut tight and upright, for template matching
    angles            coarse search angles [start, stop, step] in deg (0..180: the pill looks the same turned 180)
    model_input_size  AnomalyDINO input (height, width), multiples of 14, same ratio as the crop
    checkpoint        AnomalyDINO model written by `main.py train`
    good              folder of good photos: `main.py train` trains on them, the size check takes its nominal from them
    spec_mm           datasheet size per dimension: [nominal, +-tolerance] in mm
    hole_x            px from each crop end to a hole centre
    hole_radius       expected radius of the hole countersink edge in px
    camera            exposure (us) and gain (dB) per check, written by the app's "Save as SKU default"
"""
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import size_check
from anomaly_dino import Detector
from template_matching import TemplateMatcher

ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT / 'configs'
STEPS = [{'key': 'surface', 'name': 'Surface'}, {'key': 'dimensions', 'name': 'Dimensions'}]
PROCESS_HEIGHT = 400  # px height of each picture in the dimensions process image


# ---------- SKU config ----------

def list_skus():
    return sorted(p.stem for p in CONFIGS.glob('*.json'))


def read_config(name):
    return json.loads((CONFIGS / f'{name}.json').read_text())


def load_sku(name):
    """configs/<name>.json with the paths made absolute and the template matcher built."""
    sku = read_config(name)
    for key in ('template', 'checkpoint', 'good'):
        sku[key] = ROOT / sku[key]
    sku['name'] = name
    sku['model_input_size'] = tuple(sku['model_input_size'])
    sku['matcher'] = TemplateMatcher(sku['template'], angles=range(*sku['angles']))
    return sku


def camera_settings(name, step):
    return read_config(name)['camera'][STEPS[step]['key']]


def save_camera_settings(name, step, settings):
    config = read_config(name)
    config['camera'][STEPS[step]['key']] = settings
    (CONFIGS / f'{name}.json').write_text(json.dumps(config, indent=2) + '\n')


# ---------- drawing (surface) ----------

def pill_region(input_size, margin=12):
    """Bool mask at model size: the pill seen from above (rectangle with round ends), margin px inside the crop
    border, so the surface check skips the background in the crop corners and the pill outline."""
    h, w = input_size
    r = h // 2
    region = np.zeros((h, w), np.uint8)
    cv2.rectangle(region, (r, margin), (w - r, h - margin), 1, -1)
    for x in (r, w - r):
        cv2.circle(region, (x, r), r - margin, 1, -1)
    return region.astype(bool)


def draw_overlay(crop, result, detector):
    """Crop at model size with the heatmap blended in and the defect mask outlined in red."""
    crop = cv2.resize(crop, detector.input_size[::-1], interpolation=cv2.INTER_AREA)
    heat = np.clip(result['anomaly_map'] / (2 * detector.pixel_threshold), 0, 1)  # pixel threshold sits mid-scale
    heatmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(crop, 0.6, heatmap, 0.4, 0)
    contours, _ = cv2.findContours(result['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)


def draw_mask(crop, result):
    """Crop at full size with only the defect mask outlined in red (no heatmap)."""
    mask = cv2.resize(result['mask'].astype(np.uint8), crop.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(crop.copy(), contours, -1, (0, 0, 255), 3)


def draw_panels(crop, result, detector):
    """Crop | heatmap with mask | mask only, side by side at crop size."""
    heatmap = cv2.resize(draw_overlay(crop, result, detector), crop.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return np.hstack([crop, heatmap, draw_mask(crop, result)])


# ---------- checks ----------

def prepare(sku, step):
    """What one step needs, loaded once: the AnomalyDINO model, or the size calibration from the good photos."""
    if STEPS[step]['key'] == 'surface':
        return {'sku': sku, 'detector': Detector(sku['checkpoint'])}
    paths = size_check.image_paths(sku['good'])
    assert len(paths) >= 2, f"need at least 2 good photos in {sku['good']}"
    goods = [size_check.measure(cv2.imread(str(p)), sku) for p in paths]
    return {'sku': sku, 'calibration': size_check.calibrate(goods, sku['spec_mm'])}


def check(photo, prepared):
    if 'detector' in prepared:
        return check_surface(photo, prepared['detector'], prepared['sku'])
    return check_dimensions(photo, prepared['calibration'], prepared['sku'])


def result(faults, summary, image, process):
    return dict(verdict='NG' if faults else 'OK', faults=faults, summary=summary, image=image, process=process)


def check_surface(photo, detector, sku):
    """Result picture: the crop with the defect mask outlined; process: crop | heatmap | mask."""
    crop, _, _, match = sku['matcher'].crop(photo)
    r = detector.inspect(crop)
    pill_pixels = r['mask'].size if detector.region is None else detector.region.sum()
    faults = [f"Surface defect ({r['mask'].sum() / pill_pixels * 100:.1f}% of the pill)"] if r['defect'] else []
    summary = f"score {r['score']:.3f} / threshold {detector.image_threshold:.3f} · match {match:.2f}"
    return result(faults, summary, draw_mask(crop, r), draw_panels(crop, r, detector))


def check_dimensions(photo, cal, sku):
    """Result picture: size_check.draw_overlay; process: every per-line edge and the holes | the result."""
    r = size_check.judge(size_check.measure(photo, sku), cal)
    image = size_check.draw_overlay(photo, r, cal)
    mm = cal['mm_per_px']
    faults = [f"{d.capitalize()} {r[d] * mm:.2f} mm (nominal {cal['nominal'][d] * mm:.2f} +- {cal['limit'][d] * mm:.2f})"
              for d in r['bad']]
    summary = ' · '.join(f'{d} {r[d] * mm:.2f} mm' for d in size_check.DIMS)
    g = size_check.geometry(sku)
    edges = r['crop'].copy()  # every per-line edge the medians come from
    rows = np.arange(g['MID'] - size_check.ROWS, g['MID'] + size_check.ROWS)
    cols = np.arange(size_check.A, g['B'])
    for xs, ys in ((r['lefts'], rows), (r['rights'], rows), (cols, r['tops']), (cols, r['bots'])):
        for x, y in zip(xs, ys):
            cv2.circle(edges, (int(x), int(y)), 2, (0, 255, 255), -1)
    for h in r['holes']:
        if h:
            cv2.circle(edges, tuple(int(v) for v in h['center']), int(h['radius']), (0, 255, 0), 2)
    process = np.hstack([cv2.resize(p, (round(p.shape[1] * PROCESS_HEIGHT / p.shape[0]), PROCESS_HEIGHT))
                         for p in (edges, image)])
    return result(faults, summary, image, process)


# ---------- command line ----------

COMMANDS = {'train': 'train', 'infer': 'infer', 'size': 'size_check'}  # command -> script it runs


def main():
    """`python main.py <command> [args]` runs that script's main() with the remaining args."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__ + f"\nusage: python main.py {{{','.join(COMMANDS)}}} [args]   (add -h for a command's options)")
    module = importlib.import_module(COMMANDS[sys.argv[1]])  # here, not at the top: those scripts import this one
    sys.argv = [f'main.py {sys.argv[1]}'] + sys.argv[2:]
    module.main()


if __name__ == '__main__':
    main()
