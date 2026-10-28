"""Splitting a character sheet into the views MiniMax H3 gets as pictures.

A character sheet shows one character several times on a plain backdrop:
full-body views from several angles and often face close-ups. The swap gives H3
the front full-body view as Picture 1, then the most frontal face close-up
(small identity details such as a monocle survive only with it), the back
view, and any other views.

Figures are found by the pose estimator's person detector, and their pixels by
color distance from the backdrop, which must be plain and roughly uniform. The
keypoints tell the views apart: a full-body view shows the ankles, and the
signed widths of the shoulders and hips show which way a figure faces (the
person's left shoulder is on the image's right when they face the camera). A
photo of one person works too; it is its own front view.

Typical usage example:

  sheet = Image.open("sheet.jpg")
  views = find_views(sheet)
  crop(sheet, views, pick_front(views)).save("front.png")
"""

import itertools
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from rigless import pose

Box = tuple[int, int, int, int]  # left, top, right, bottom

FOUND = 0.5  # keypoint confidence; RTMPose guesses unseen points at up to ~0.45
BACKDROP_DISTANCE = 30  # RGB distance from the backdrop color that counts as figure
BACK = -0.25  # facing() below this is a back or three-quarter back view
MARGIN = 0.04  # of the figure's height, kept around a crop

NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


# Compared and hashed by identity: two figures are never the same one.
@dataclass(frozen=True, eq=False)
class View:
    """One figure on a character sheet.

    Attributes:
        box: The figure's tight (left, top, right, bottom) box in sheet pixels.
        mask: HxW bool array, True on the figure's pixels.
        keypoints: (17, 2) COCO keypoint positions in sheet pixels.
        scores: (17,) keypoint confidences.
    """

    box: Box
    mask: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray

    @property
    def full_body(self) -> bool:
        """True if the view shows the whole body, ankles included."""
        return bool(self.scores[[LEFT_ANKLE, RIGHT_ANKLE]].min() > FOUND)


@dataclass(frozen=True)
class Views:
    """The views of a character, cut out of its images.

    Attributes:
        front: The full-body view that faces the camera most squarely.
        close_up: The most frontal face close-up, or None if there is none.
        back: The full-body view from behind, or None if there is none.
        warning: Advice about the front view's pose, or None.
        others: Every other view, in the order found.
    """

    front: Image.Image
    close_up: Image.Image | None
    back: Image.Image | None
    warning: str | None
    others: tuple[Image.Image, ...] = ()


def upright(image: Image.Image) -> Image.Image:
    """Returns an RGB copy of the image as it's shown, on white where transparent.

    A phone photo is often stored sideways with a tag saying how to turn it;
    a cutout's transparent backdrop would otherwise read as whatever color
    the hidden pixels hold, often black.
    """
    turned = ImageOps.exif_transpose(image)
    if "A" not in turned.getbands() and "transparency" not in turned.info:
        return turned.convert("RGB")
    rgba = turned.convert("RGBA")
    flat = Image.new("RGB", rgba.size, "white")
    flat.paste(rgba, mask=rgba.getchannel("A"))
    return flat


def extract(images: list[Image.Image]) -> Views:
    """Picks and cuts out the views, pooled across a sheet or several images.

    Raises:
        ValueError: No image shows the character's whole body.
    """
    found = [(image, find_views(image)) for image in images]
    owners = {view: (image, views) for image, views in found for view in views}
    pooled = [view for _, views in found for view in views]

    def cut(view: View | None) -> Image.Image | None:
        if view is None:
            return None
        image, views = owners[view]
        return crop(image, views, view)

    front = pick_front(pooled)
    close_up = pick_close_up(pooled)
    back = pick_back(pooled)
    picked = {view for view in (front, close_up, back) if view is not None}
    return Views(
        crop(*owners[front], front),
        cut(close_up),
        cut(back),
        pose_warning(front),
        tuple(crop(*owners[view], view) for view in pooled if view not in picked),
    )


def find_views(sheet: Image.Image) -> list[View]:
    """Finds every figure on a character sheet, on the CPU.

    Detections that cover no foreground pixels are dropped.
    """
    image = np.asarray(sheet.convert("RGB"))
    people = pose.find_people(image)
    masks = split(
        foreground(image),
        [person.keypoints[person.scores > FOUND] for person in people],
    )
    return [
        View(_bounds(mask), mask, person.keypoints, person.scores)
        for person, mask in zip(people, masks, strict=True)
        if mask.any()
    ]


def foreground(image: np.ndarray) -> np.ndarray:
    """Returns an HxW bool mask of the pixels that differ from the backdrop.

    Args:
        image: An HxWx3 RGB image whose border is mostly backdrop.
    """
    distance = np.linalg.norm(image.astype(np.float32) - _backdrop(image), axis=2)
    mask = (distance > BACKDROP_DISTANCE).astype(np.uint8)
    # Opening drops the JPEG speckle that hovers around the threshold.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def split(mask: np.ndarray, keypoints: list[np.ndarray]) -> list[np.ndarray]:
    """Splits a foreground mask into one mask per figure.

    Each connected blob goes to the figure whose keypoints fall on it, and blobs
    with no keypoints (text, stray marks) are dropped. A blob that several
    figures share, such as close-ups whose shoulders touch, is cut between each
    pair of neighbors at the column (or, for figures above one another, the
    row) that crosses the fewest of its pixels: where they touch.

    Args:
        mask: HxW bool foreground mask.
        keypoints: Each figure's confident keypoints, as an (n, 2) array.

    Returns:
        One HxW bool mask per figure, in the order of keypoints.
    """
    _, labels = cv2.connectedComponents(mask.astype(np.uint8))
    height, width = mask.shape
    owners: dict[int, list[int]] = {}
    for figure, points in enumerate(keypoints):
        xs = np.clip(points[:, 0].astype(int), 0, width - 1)
        ys = np.clip(points[:, 1].astype(int), 0, height - 1)
        for label in set(labels[ys, xs].tolist()) - {0}:
            owners.setdefault(label, []).append(figure)

    masks = [np.zeros_like(mask) for _ in keypoints]
    for label, figures in owners.items():
        blob = labels == label
        pieces = _cut(blob, [keypoints[figure] for figure in figures])
        for figure, piece in zip(figures, pieces, strict=True):
            masks[figure] |= piece
    return masks


def facing(keypoints: np.ndarray) -> float:
    """Returns how squarely a full-body figure faces the camera.

    The signed widths of the shoulders and hips over the torso's length: about
    0.5 for a front view, near 0 in profile, and negative from behind.
    """
    k = keypoints
    shoulders = k[LEFT_SHOULDER, 0] - k[RIGHT_SHOULDER, 0]
    hips = k[LEFT_HIP, 0] - k[RIGHT_HIP, 0]
    torso = (
        k[[LEFT_HIP, RIGHT_HIP], 1].mean()
        - k[[LEFT_SHOULDER, RIGHT_SHOULDER], 1].mean()
    )
    return float((shoulders + hips) / (2 * torso))


def face_score(keypoints: np.ndarray) -> float:
    """Rates a close-up as an identity reference: a big face, seen from the front.

    The gap between the eyes in pixels, discounted as the nose moves off center
    between them, as it does when the head turns. Negative from behind.
    """
    gap = keypoints[LEFT_EYE, 0] - keypoints[RIGHT_EYE, 0]
    middle = keypoints[[LEFT_EYE, RIGHT_EYE], 0].mean()
    off_center = abs(keypoints[NOSE, 0] - middle) / max(abs(gap), 1)
    return float(gap * max(0.0, 1 - off_center))


def pick_front(views: list[View]) -> View:
    """Returns the full-body view that faces the camera most squarely.

    Raises:
        ValueError: No view shows the whole body.
    """
    full_body = [view for view in views if view.full_body]
    if not full_body:
        raise ValueError(
            "No full-body view found. The sheet needs a head-to-toe view of the "
            "character, facing the camera."
        )
    return max(full_body, key=lambda view: facing(view.keypoints))


def pick_back(views: list[View]) -> View | None:
    """Returns the full-body view seen most squarely from behind, if any."""
    backs = [v for v in views if v.full_body and facing(v.keypoints) < BACK]
    return min(backs, key=lambda view: facing(view.keypoints), default=None)


def pick_close_up(views: list[View]) -> View | None:
    """Returns the close-up that shows the face biggest and most frontally, if any."""
    close_ups = [v for v in views if not v.full_body and face_score(v.keypoints) > 0]
    return max(close_ups, key=lambda view: face_score(view.keypoints), default=None)


def pose_warning(view: View) -> str | None:
    """Returns a warning if a front view's hands rest on or cross the torso.

    Frozen takes fall back to the reference's pose, and hands clasped in front
    of the body is the pose they fell back to in testing.
    """
    k, s = view.keypoints, view.scores
    middle = k[[LEFT_SHOULDER, RIGHT_SHOULDER], 0].mean()
    shoulder_y = k[[LEFT_SHOULDER, RIGHT_SHOULDER], 1].mean()
    for shoulder, wrist in ((LEFT_SHOULDER, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_WRIST)):
        reach = k[shoulder, 0] - middle
        if s[wrist] < FOUND or k[wrist, 1] < shoulder_y or abs(reach) < 1:
            continue
        # 1 below its own shoulder, 0 on the body's midline, negative past it.
        if (k[wrist, 0] - middle) / reach < 0.5:
            return (
                "The character's hands rest on the body in the front view. Takes "
                "follow the dance better from a reference with the arms held a "
                "little away from the body."
            )
    return None


def crop(sheet: Image.Image, views: list[View], view: View) -> Image.Image:
    """Cuts one view out of the sheet with a margin of its backdrop.

    Parts of other figures that reach into the margin are painted over with the
    backdrop color.
    """
    image = np.asarray(sheet.convert("RGB")).copy()
    others = np.zeros(view.mask.shape, np.uint8)
    for other in views:
        if other is not view:
            others |= other.mask
    # Widened so their soft, below-threshold edges go too.
    widened = cv2.dilate(others, np.ones((7, 7), np.uint8)) > 0
    image[widened & ~view.mask] = _backdrop(image)

    left, top, right, bottom = view.box
    pad = round(MARGIN * (bottom - top))
    height, width = view.mask.shape
    return Image.fromarray(image).crop(
        (
            max(left - pad, 0),
            max(top - pad, 0),
            min(right + pad, width),
            min(bottom + pad, height),
        )
    )


def cutout(sheet: Image.Image, view: View) -> Image.Image:
    """Cuts one view out of the sheet as RGBA, transparent off the figure.

    Parts of the figure as gray as the backdrop, such as a white shirt in
    shadow, slip through the color threshold as thin seams and small holes; they
    are closed and filled. The edge is then pulled in by a pixel and softened,
    so no backdrop fringe shows when the figure is pasted onto a new background.
    """
    closed = cv2.morphologyEx(
        view.mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    # Real gaps, such as between an arm and the torso, are larger.
    mask = _fill_holes(closed, max_area=0.01 * int(closed.sum()))
    alpha = cv2.erode(mask * 255, np.ones((3, 3), np.uint8))
    image = sheet.convert("RGBA")
    image.putalpha(Image.fromarray(cv2.GaussianBlur(alpha, (0, 0), 0.8)))
    return image.crop(view.box)


def _fill_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Fills a 0/1 mask's enclosed holes that are smaller than max_area pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - mask, connectivity=4)
    height, width = mask.shape
    small = [
        label
        for label, (x, y, w, h, area) in enumerate(stats)
        if label > 0
        and x > 0
        and y > 0
        and x + w < width
        and y + h < height
        and area < max_area
    ]
    return mask | np.isin(labels, small).astype(np.uint8)


def _backdrop(image: np.ndarray) -> np.ndarray:
    """The backdrop's color: the median of the image's outermost pixels."""
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    return np.median(border, axis=0)


def _cut(blob: np.ndarray, keypoints: list[np.ndarray]) -> list[np.ndarray]:
    """Cuts a blob into one piece per figure; see `split`."""
    centers = np.array([points.mean(axis=0) for points in keypoints])
    axis = int(np.ptp(centers[:, 1]) > np.ptp(centers[:, 0]))  # 0: x, 1: y
    counts = blob.sum(axis=axis)  # blob pixels per column (x) or per row (y)
    order = np.argsort(centers[:, axis])
    cuts = []
    for a, b in itertools.pairwise(order):
        low, high = int(centers[a, axis]), int(centers[b, axis])
        cuts.append(low + int(np.argmin(counts[low : high + 1])))
    position = np.indices(blob.shape)[1 - axis]
    limits = [0, *cuts, blob.shape[1 - axis]]
    pieces = [blob] * len(keypoints)
    for i, figure in enumerate(order):
        pieces[figure] = blob & (position >= limits[i]) & (position < limits[i + 1])
    return pieces


def _bounds(mask: np.ndarray) -> Box:
    """The (left, top, right, bottom) box around a mask's True pixels."""
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
