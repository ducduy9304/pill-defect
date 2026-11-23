"""AnomalyDINO surface check on images that are already cropped to the part (BGR, as OpenCV reads them).

Standalone: needs torch, anomalib, OpenCV, NumPy. Copy this file into another project; crop the part first
(for example with template_matching.py), all crops the same size.

    import anomaly_dino
    checkpoint = anomaly_dino.train(good_crops, input_size=(252, 742))   # (height, width), multiples of 14
    torch.save(checkpoint, 'model.pt')
    detector = anomaly_dino.Detector('model.pt')
    result = detector.inspect(crop)    # dict(defect, score, anomaly_map, mask)

Model: DINOv2 ViT-S/14 patch features, score = cosine distance to the nearest good patch (memory bank).
Thresholds by leave-one-out on the good crops: for each one, memory bank = the others' features, score it.
Image threshold = max leave-one-out score x image_margin, pixel threshold = max leave-one-out pixel x pixel_margin.
The final memory bank uses all good crops.
Optional region (bool mask at input size, e.g. the part without the background around it): score, mask and both
thresholds use only the anomaly map inside it, score = mean of its top 1% pixels. It is saved in the checkpoint.
"""
import numpy as np
import torch
from anomalib.models import AnomalyDINO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_tensor(crops, pre_processor):
    """BGR crops -> pre-processed batch (resize + ImageNet normalize) on DEVICE."""
    rgb_batch = np.stack([crop[..., ::-1].copy() for crop in crops])
    batch = torch.from_numpy(rgb_batch).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    return pre_processor.transform(batch)


def scores(output, region):
    """(image score, anomaly map): the model's own, or the top 1% mean and the map limited to region."""
    anomaly_map = output.anomaly_map[0, 0].cpu().numpy()
    if region is None:
        return output.pred_score[0, 0].item(), anomaly_map
    anomaly_map = anomaly_map * region
    inside = np.sort(anomaly_map[region])
    return float(inside[-max(len(inside) // 100, 1):].mean()), anomaly_map


@torch.no_grad()
def train(crops, input_size, region=None, image_margin=1.2, pixel_margin=1.1, log=print):
    """Good BGR crops -> checkpoint dict (memory bank, thresholds, input size). Save it with torch.save."""
    assert len(crops) >= 2, 'need at least 2 good crops for leave-one-out'
    pre_processor = AnomalyDINO.configure_pre_processor(image_size=input_size)
    model = AnomalyDINO().model.to(DEVICE)
    model.train()  # training mode = forward appends patch features to embedding_store
    for crop in crops:
        model(to_tensor([crop], pre_processor))
    features_per_image = list(model.embedding_store)  # one [patches, 384] tensor per crop
    model.embedding_store.clear()

    model.eval()
    left_out_scores, left_out_max_pixels = [], []
    for index, crop in enumerate(crops):
        model.memory_bank = torch.cat(features_per_image[:index] + features_per_image[index + 1:])
        score, anomaly_map = scores(model(to_tensor([crop], pre_processor)), region)
        left_out_scores.append(score)
        left_out_max_pixels.append(float(anomaly_map.max()))
        log(f'crop {index}: leave-one-out score {left_out_scores[-1]:.4f}, max pixel {left_out_max_pixels[-1]:.4f}')

    return dict(memory_bank=torch.cat(features_per_image).cpu(), input_size=tuple(input_size),
                region=None if region is None else torch.from_numpy(region),
                image_threshold=float(np.max(left_out_scores) * image_margin),
                pixel_threshold=float(np.max(left_out_max_pixels) * pixel_margin))


class Detector:
    """Loads a checkpoint written by train() once, then inspects crops."""

    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.input_size = checkpoint['input_size']
        self.image_threshold = checkpoint['image_threshold']
        self.pixel_threshold = checkpoint['pixel_threshold']
        region = checkpoint.get('region')  # older checkpoints have none: the whole crop
        self.region = None if region is None else region.cpu().numpy()
        self.pre_processor = AnomalyDINO.configure_pre_processor(image_size=self.input_size)
        self.model = AnomalyDINO().model.to(DEVICE)
        self.model.memory_bank = checkpoint['memory_bank']
        self.model.eval()

    @torch.no_grad()
    def inspect(self, crop):
        """BGR crop -> dict(defect, score, anomaly_map, mask). anomaly_map / mask are at input_size.
        Defect when the score or any pixel passes its threshold, so a drawn mask always means NG."""
        score, anomaly_map = scores(self.model(to_tensor([crop], self.pre_processor)), self.region)
        mask = anomaly_map > self.pixel_threshold
        return dict(defect=score > self.image_threshold or bool(mask.any()), score=score,
                    anomaly_map=anomaly_map, mask=mask)
