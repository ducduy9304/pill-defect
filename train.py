"""Build the AnomalyDINO checkpoint of a SKU: crop its good photos with its template matcher, then
anomaly_dino.train (see its docstring).

    python train.py                                    # SKU sku01: its good photos -> its checkpoint
    python train.py --sku sku01 --good some/dir --out models/x.pt
    python main.py train ...                           # same thing
"""
import argparse
from pathlib import Path

import cv2
import torch

import anomaly_dino
import size_check
from main import load_sku, pill_region


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sku', default='sku01', help='SKU config in configs/ (template, model input size)')
    parser.add_argument('--good', help="folder of good raw images, default: the SKU's good")
    parser.add_argument('--out', help="checkpoint to write, default: the SKU's checkpoint")
    args = parser.parse_args()

    sku = load_sku(args.sku)
    out = Path(args.out or sku['checkpoint'])
    crops = []
    for index, path in enumerate(size_check.image_paths(args.good or sku['good'])):
        image = cv2.imread(str(path))
        assert image is not None, path
        crop, _, angle, match_score = sku['matcher'].crop(image)
        crops.append(crop)
        print(f'crop {index} = {path.name}: match {match_score:.2f}, angle {angle:.0f}')

    checkpoint = anomaly_dino.train(crops, sku['model_input_size'], pill_region(sku['model_input_size']))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    print(f"image threshold {checkpoint['image_threshold']:.4f}, "
          f"pixel threshold {checkpoint['pixel_threshold']:.4f}, "
          f"memory bank {tuple(checkpoint['memory_bank'].shape)} -> {out}")


if __name__ == '__main__':
    main()
