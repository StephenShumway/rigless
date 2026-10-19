"""Composing the reference image: the character edited into a background photo.

Wan Animate 2 carries its one reference image's background into the video, so to
show the character in a real place, the reference must already show them there.
Qwen-Image-Edit 2511 does the edit and matches the scene's lighting, which a plain
paste can't. Its workflow takes the background as Picture 1 (which also sets the
output canvas), the character's full-body view as Picture 2, and an optional face
close-up as Picture 3; the prompt refers to them by those names.

In a full-body shot the face is small, so details such as a monocle come out
wrong. The face-detail pass fixes them: cut the head out with `face_box`, upscale
it, re-edit it with the same workflow (the crop as Picture 1, the sheet's face
close-up as Picture 2), and put the result back with `blend_face`.

Typical usage example:

  fit_background(Image.open(photo), width=720, height=1280).save(canvas)
  graph = patch(template, background=canvas_asset, character=front_asset,
                close_up=None, prompt="Replace the man in Picture 1 with ...",
                seed=1)
"""

import copy

import numpy as np
from comfy_sdk import Asset
from PIL import Image, ImageDraw, ImageFilter

from rigless.workflow import Graph, find_node

Box = tuple[int, int, int, int]  # left, top, right, bottom


def fit_background(photo: Image.Image, width: int, height: int) -> Image.Image:
    """Fits a background photo to the output's aspect ratio for the editor.

    A photo wider than the output keeps its full height and gains a plain band
    at the bottom, up to a quarter of the result, for the editor to fill with
    floor so the character fits head to shoes; any width still left over is
    cropped evenly from both sides. A narrower photo is cropped evenly top and
    bottom.

    Args:
        photo: The background photo.
        width: The output video's width; only the ratio to height matters.
        height: The output video's height.

    Returns:
        A new RGB image with the output's aspect ratio, to within a pixel.
    """
    photo = photo.convert("RGB")
    aspect = width / height
    w, h = photo.size
    if w / h <= aspect:
        crop_h = round(w / aspect)
        top = (h - crop_h) // 2
        return photo.crop((0, top, w, top + crop_h))
    canvas_h = min(round(w / aspect), round(h * 4 / 3))
    canvas_w = round(canvas_h * aspect)
    # The band starts as the floor's average color, a better hint than gray.
    floor = photo.crop((0, h - 20, w, h)).resize((1, 1), Image.Resampling.BOX)
    canvas = Image.new("RGB", (canvas_w, canvas_h), floor.getpixel((0, 0)))
    canvas.paste(photo, ((canvas_w - w) // 2, 0))
    return canvas


def patch(
    graph: Graph,
    *,
    background: str | Asset,
    character: str | Asset,
    close_up: str | Asset | None,
    prompt: str,
    seed: int,
    full_sampling: bool = False,
) -> Graph:
    """Returns a copy of the compose workflow set up for one image.

    Args:
        graph: The workflow template. It is not modified.
        background: The fitted background photo (Picture 1).
        character: The character's full-body front view (Picture 2).
        close_up: A close-up of the character's face (Picture 3), or None to
            leave Picture 3 out.
        prompt: The edit instruction, naming the inputs Picture 1 to 3.
        seed: The sampler's noise seed.
        full_sampling: Sample without the 4-step Lightning LoRA, at 40 steps and
            cfg 4 (the template's defaults). About 20x slower, but softer and
            more photographic, a look the video then inherits.
    """
    patched = copy.deepcopy(graph)
    background_id = find_node(patched, "LoadImage", "Background")
    character_id = find_node(patched, "LoadImage", "Character")
    close_up_id = find_node(patched, "LoadImage", "Character close-up")
    patched[background_id]["inputs"]["image"] = background
    patched[character_id]["inputs"]["image"] = character
    if close_up is None:
        del patched[close_up_id]
        for node in patched.values():
            node["inputs"] = {
                name: value
                for name, value in node["inputs"].items()
                if value != [close_up_id, 0]
            }
    else:
        patched[close_up_id]["inputs"]["image"] = close_up
    positive = find_node(
        patched,
        "TextEncodeQwenImageEditPlus",
        "TextEncodeQwenImageEditPlus (Positive)",
    )
    patched[positive]["inputs"]["prompt"] = prompt
    sampler = patched[find_node(patched, "KSampler")]["inputs"]
    sampler["seed"] = seed
    if full_sampling:
        sampler["steps"], sampler["cfg"] = 40, 4
        lora = find_node(patched, "LoraLoaderModelOnly")
        patched[lora]["inputs"]["strength_model"] = 0
    return patched


def face_box(points: np.ndarray, image_size: tuple[int, int]) -> Box:
    """Returns a square box around the head for the face-detail pass.

    The box spans about three ear-widths, sits a little above the face's center
    so it takes in the hair and the collar, and is shifted to lie inside the
    image.

    Args:
        points: The face keypoints from `pose.face_keypoints`.
        image_size: The image's (width, height).

    Returns:
        The box as (left, top, right, bottom) pixels.
    """
    width, height = image_size
    cx, cy = points.mean(axis=0)
    side = int(min(max(np.ptp(points[:, 0]) * 3.2, 128), width, height))
    left = int(np.clip(cx - side / 2, 0, width - side))
    top = int(np.clip(cy - side * 0.45, 0, height - side))
    return left, top, left + side, top + side


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Returns the scale, rotation, and shift that best map src points onto dst.

    A least-squares fit (Umeyama), so one badly placed keypoint can't skew it.

    Args:
        src: (n, 2) points.
        dst: (n, 2) matching points.

    Returns:
        A 2x3 matrix M with dst ≈ M @ [x, y, 1].
    """
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    u, sigma, vt = np.linalg.svd(dst_c.T @ src_c)
    sign = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])  # a rotation, not a flip
    rotation = u @ sign @ vt
    scale = (sigma * np.diag(sign)).sum() / (src_c**2).sum()
    shift = dst_mean - scale * rotation @ src_mean
    return np.column_stack([scale * rotation, shift])


def blend_face(
    reference: Image.Image,
    box: Box,
    points: np.ndarray,
    edit: Image.Image,
    edit_points: np.ndarray,
) -> Image.Image:
    """Returns the reference with the re-edited face blended in.

    The editor tends to zoom in and swap the backdrop, so the edit is first
    aligned to the reference by its face keypoints, and only the inner face
    (temples to chin) is blended, with its colors matched to the reference's.
    Hair, ears, and everything outside the face keep the reference's pixels.

    Args:
        reference: The full reference image.
        box: The face box the edit was made from (see `face_box`).
        points: The face keypoints in the reference.
        edit: The re-edited crop of box, at any size.
        edit_points: The face keypoints in the edit.
    """
    left, top, right, bottom = box
    side = right - left
    target = (points - (left, top)) * (edit.width / side)
    matrix = similarity_transform(edit_points, target)
    inverse = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2]  # PIL maps out -> in
    aligned = edit.convert("RGB").transform(
        edit.size,
        Image.Transform.AFFINE,
        tuple(inverse.ravel()),
        Image.Resampling.BICUBIC,
    )
    aligned = aligned.resize((side, bottom - top), Image.Resampling.LANCZOS)
    crop = reference.convert("RGB").crop(box)
    mask = _face_oval(points - (left, top), crop.size)
    result = reference.convert("RGB")
    result.paste(
        _match_colors(aligned, crop, np.asarray(mask) > 128), (left, top), mask
    )
    return result


def _face_oval(points: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """A soft-edged mask of the inner face: temples to chin, below the hairline."""
    eyes = points[1:3]
    gap = float(np.linalg.norm(eyes[0] - eyes[1]))
    x, y = eyes.mean(axis=0)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse(
        (x - 1.05 * gap, y - 1.1 * gap, x + 1.05 * gap, y + 1.75 * gap), fill=255
    )
    return mask.filter(ImageFilter.GaussianBlur(0.12 * gap))


def _match_colors(
    image: Image.Image, target: Image.Image, region: np.ndarray
) -> Image.Image:
    """Shifts image's colors so its mean and spread match target's inside region."""
    a = np.asarray(image, dtype=np.float32)
    b = np.asarray(target, dtype=np.float32)
    a_in, b_in = a[region], b[region]
    a = (a - a_in.mean(axis=0)) / (a_in.std(axis=0) + 1e-6) * b_in.std(axis=0)
    return Image.fromarray(np.clip(a + b_in.mean(axis=0), 0, 255).astype(np.uint8))
