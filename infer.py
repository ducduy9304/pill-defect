"""Surface check of raw photos: print the score, save crop | heatmap | mask per photo.

    python infer.py data/raw/defect/*.bmp              # result images in outputs/infer
    python infer.py data/raw/defect/1.bmp --sku sku01 --checkpoint models/pill_dino.pt --out outputs/infer
    python main.py infer ...                           # same thing
"""
import argparse
from pathlib import Path

import cv2

from anomaly_dino import Detector
from main import draw_panels, load_sku


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', nargs='+')
    parser.add_argument('--sku', default='sku01', help='SKU config in configs/ (template, checkpoint)')
    parser.add_argument('--checkpoint', help="AnomalyDINO model, default: the SKU's checkpoint")
    parser.add_argument('--out', default='outputs/infer', help='folder for result images (crop | heatmap | mask)')
    args = parser.parse_args()

    sku = load_sku(args.sku)
    detector = Detector(args.checkpoint or sku['checkpoint'])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in args.images:
        image = cv2.imread(image_path)
        if image is None:
            print(f'{image_path}: cannot read, skipped')
            continue
        crop, _, _, match_score = sku['matcher'].crop(image)
        r = detector.inspect(crop)
        label = 'DEFECT' if r['defect'] else 'good'
        print(f"{image_path}: {label}  score {r['score']:.4f} (threshold {detector.image_threshold:.4f})  "
              f"mask {r['mask'].mean() * 100:.1f}%  crop match {match_score:.2f}")
        out_name = f'{Path(image_path).parent.name}_{Path(image_path).stem}.png'
        cv2.imwrite(str(out_dir / out_name), draw_panels(crop, r, detector))


if __name__ == '__main__':
    main()
