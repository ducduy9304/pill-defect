"""Find a part in an image by rotated template matching, then cut it out upright (template size, plus a margin).

Standalone: needs only OpenCV and NumPy. Copy this file into another project and give it that project's template.

    from template_matching import TemplateMatcher
    matcher = TemplateMatcher('template.bmp')                        # part not symmetric: angles 0..355
    matcher = TemplateMatcher('template.bmp', angles=range(0, 180, 5))   # part looks the same turned 180 deg
    crop, center, angle, score = matcher.crop(cv2.imread('raw.bmp'), pad=20)

Coarse to fine: every angle on a 1/16 image, then +-fine_angle_span deg in 1 deg steps on a 1/4 image,
only in a window around the coarse hit. score = normalized cross-correlation (1 = perfect match).
"""
import cv2
import numpy as np


def to_gray(bgr_image, scale):
    resized = cv2.resize(bgr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def rotate_on_canvas(image, angle):
    """Rotate around the center on a canvas big enough to keep every pixel. Returns image and 2x3 matrix."""
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    cos, sin = abs(rotation[0, 0]), abs(rotation[0, 1])
    canvas_width = int(width * cos + height * sin)
    canvas_height = int(width * sin + height * cos)
    rotation[:, 2] += (canvas_width / 2 - width / 2, canvas_height / 2 - height / 2)
    rotated = cv2.warpAffine(image, rotation, (canvas_width, canvas_height), borderMode=cv2.BORDER_REPLICATE)
    return rotated, rotation


def search_angles(gray_image, gray_template, angles):
    """Best (match score, angle, part center in gray_image coords) over the given angles."""
    template_height, template_width = gray_template.shape
    best = (-1, 0, (0, 0))
    for angle in angles:
        rotated, rotation = rotate_on_canvas(gray_image, angle)
        match_map = cv2.matchTemplate(rotated, gray_template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (left, top) = cv2.minMaxLoc(match_map)
        if score > best[0]:
            center_in_rotated = (left + template_width / 2, top + template_height / 2, 1)
            center_x, center_y = cv2.invertAffineTransform(rotation) @ center_in_rotated
            best = (score, angle, (center_x, center_y))
    return best


class TemplateMatcher:
    def __init__(self, template, angles=range(0, 360, 5), coarse_scale=1 / 16, fine_scale=0.25, fine_angle_span=4):
        """template: path or BGR image of the part, upright, cut tight. angles: coarse search angles (deg)."""
        if not isinstance(template, np.ndarray):
            path, template = template, cv2.imread(str(template))
            assert template is not None, f'cannot read template {path}'
        self.height, self.width = template.shape[:2]
        self.angles = np.asarray(angles)
        self.coarse_scale, self.fine_scale, self.fine_angle_span = coarse_scale, fine_scale, fine_angle_span
        self.coarse_template = to_gray(template, coarse_scale)
        self.fine_template = to_gray(template, fine_scale)

    def find(self, image):
        """BGR image -> (part center in full-res px, angle deg, match score)."""
        _, angle, (center_x, center_y) = search_angles(to_gray(image, self.coarse_scale), self.coarse_template, self.angles)

        # refine only in a window around the coarse hit: 5x faster than the whole image
        scale_up = self.fine_scale / self.coarse_scale
        center_x, center_y = center_x * scale_up, center_y * scale_up
        half_window = int(max(self.fine_template.shape) * 0.75)  # the template at any angle plus coarse error
        fine_image = to_gray(image, self.fine_scale)
        window_left = max(int(center_x) - half_window, 0)
        window_top = max(int(center_y) - half_window, 0)
        window = fine_image[window_top:int(center_y) + half_window, window_left:int(center_x) + half_window]
        fine_angles = np.arange(angle - self.fine_angle_span, angle + self.fine_angle_span + 1, 1)
        score, angle, (window_x, window_y) = search_angles(window, self.fine_template, fine_angles)

        center = ((window_x + window_left) / self.fine_scale, (window_y + window_top) / self.fine_scale)
        return center, angle, score

    def crop_matrix(self, center, angle, pad=0):
        """2x3 matrix raw image -> crop (template size plus pad px on every side)."""
        rotation = cv2.getRotationMatrix2D(center, angle, 1)
        rotation[:, 2] += (self.width / 2 + pad - center[0], self.height / 2 + pad - center[1])  # part center -> crop center
        return rotation

    def crop(self, image, pad=0):
        """Cut the part out upright, template size plus pad px on every side.
        Returns (crop, center, angle, match score)."""
        center, angle, score = self.find(image)
        size = (self.width + 2 * pad, self.height + 2 * pad)
        crop = cv2.warpAffine(image, self.crop_matrix(center, angle, pad), size, borderMode=cv2.BORDER_REPLICATE)
        return crop, center, angle, score
