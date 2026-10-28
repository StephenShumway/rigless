"""Character swap with MiniMax H3: the tracked person redrawn as the character.

H3 redraws only the tracked person, inside their mask grown by a margin, in
overlapping windows, with the clip as <Video 1> and the character's pictures as
<Picture 1> to <Picture N>. Everything outside the mask keeps the clip's own
latents, so the scene and the other people stay as filmed. The graph is built
from ComfyUI's MiniMax H3 nodes and the H3 Relay node pack (GPL-3.0, installed
separately in ComfyUI's custom_nodes), with the Rigless LoRA.

H3 hears the clip silent, so it doesn't lip-sync to the soundtrack; each take
gets the clip's own sound back afterward.

Typical usage example:

  takes = list(render_takes(part, [masks], pictures, prompt=prompt([...]),
                            seeds=[1, 2], out_dir=run_dir,
                            sound=(dance, 2.0)))
  print([take.score for take in takes])
"""

import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from comfy_sdk import Asset, Progress

from rigless import comfy_client, pose, video_prep
from rigless.comfy_client import Graph
from rigless.score import TakeScore, score_take

MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
SWAP_LORA = "rigless-lora-v1.safetensors"  # step 800 of the Oct 25 run
FPS = 24.0
STEPS = 8
SAMPLER = "sample"  # the node whose steps take most of a take's time
RESOLUTIONS = ("480", "768")  # H3 Relay's budgets: about 0.5 and 1 megapixels
# How far the mask grows, in H3's pixels at 768p, to leave room for hair and
# clothes: the Rigless LoRA trained at this margin (40 px at 480p), as did
# H3 Relay's v1.5. It and the paste's margins scale with the resolution.
EXPAND = 64
WINDOW_FRAMES = 124
HISTORY_FRAMES = 18  # frames each window carries over from the one before
MAX_PICTURES = 9
# H3 draws a take's first few frames more from the pictures than from the clip:
# Shoot 1 opens from behind, and its take faced the camera for 3 frames. So
# the clip goes in led by its own first frames played backward, and those are
# cut from the take; with 8, the turn happened inside them (Oct 27).
LEAD_IN = 8
# H3 Relay redraws whole 32-pixel cells around the mask, unioned over 4-frame
# spans, so the paste reaches that far past the mask, in H3's pixels and frames.
PASTE_PAD = 32
PASTE_SPAN = 3
FEATHER = 8.0  # the soft edge where H3's take meets the footage, in H3's pixels
# Every take says it's AI-made, in its metadata, as H3's license encourages.
AI_LABEL = "AI-generated: the person was redrawn by MiniMax H3, with Rigless."

# The prompt the Rigless LoRA was trained with, for one picture.
_REPLACE = "Replace the selected person in <Video 1> with the person in <Picture 1>."
_KEEP = (
    "Keep the reference person's face, hair and clothing. Follow the source "
    "performance's pose, gaze, expression and movement. Preserve the scene and "
    "the other people."
)
_ROLES = {
    "face": "<Picture {n}> is a close-up of the same person's face.",
    "back": "<Picture {n}> shows them from behind.",
}
_OTHER_VIEW = "<Picture {n}> is another view of them."
_GROUP_KEEP = (
    "Keep each reference person's face, hair and clothing. Follow each source "
    "performance's pose, gaze, expression and movement. Preserve the scene and "
    "the other people."
)
_TAG = re.compile(r"@(\d)")  # "@2" in the user's notes: the second person's character


@dataclass(frozen=True)
class Casting:
    """One of several people to replace, and the character they become.

    Attributes:
        where: Where they are in the clip, such as "on the left".
        roles: Their character's pictures' roles, in order, as for prompt().
        look: What they wear in the clip, such as "in a red jacket", if known.
        description: Their character's appearance in words, if any.
        unlike: Things they have that their character doesn't.
    """

    where: str
    roles: tuple[str, ...]
    look: str = ""
    description: str = ""
    unlike: tuple[str, ...] = ()


@dataclass(frozen=True)
class Take:
    """One rendered or failed take.

    Attributes:
        seed: The sampler's noise seed.
        dir: The folder holding the take's videos and run.json.
        error: Why the take failed, or None if it succeeded.
        score: How faithfully the take follows the performance, or None if it
            failed.
    """

    seed: int
    dir: Path
    error: str | None = None
    score: TakeScore | None = None

    @property
    def result(self) -> Path:
        """H3's render, as scored."""
        return self.dir / "result.mp4"

    @property
    def finished(self) -> Path:
        """The render with the clip's sound, the one to show."""
        return self.dir / "finished.mp4"


def _rank(score: TakeScore) -> tuple[bool, float]:
    """Orders scores so that the best is the largest, flagged ones last."""
    return not (score.frozen or score.mirrored), score.score


def prompt(
    roles: Sequence[str],
    notes: str = "",
    *,
    description: str = "",
    unlike: Sequence[str] = (),
) -> str:
    """Returns H3's prompt, saying what each picture shows.

    Args:
        roles: Each picture's role, <Picture 1> first: "front", "face",
            "back", or "view". The first is taken as the main reference; it's
            the back view when the clip opens with the person facing away.
        notes: The user's own words about the character, if any; "@1" in
            them means the character.
        description: The character's appearance in words, if any.
        unlike: Things the replaced person has that the character doesn't,
            such as "rings", which H3 would otherwise carry over.
    """
    sentences = [_REPLACE, *_views(roles, 1), _KEEP]
    if description.strip():
        sentences.append(f"The person in the pictures: {description.strip()}")
    if unlike:
        sentences.append(f"Unlike the original person, they have {_none_of(unlike)}.")
    if notes.strip():
        sentences.append(_tagged(notes.strip(), [1]))
    return " ".join(sentences)


def group_prompt(castings: Sequence[Casting], notes: str = "") -> str:
    """Returns H3's prompt for replacing several people at once.

    Each casting's pictures follow the previous one's: the first casting's
    are <Picture 1> on, the next one's after them.

    Args:
        castings: The people, in the order their pictures are given.
        notes: The user's own words, if any; "@2" in them means the second
            person's character.
    """
    firsts = [1]
    for casting in castings[:-1]:
        firsts.append(firsts[-1] + len(casting.roles))
    sentences = [f"Replace the {len(castings)} selected people in <Video 1>."]
    for first, casting in zip(firsts, castings, strict=True):
        who = f"{casting.where}, {casting.look}," if casting.look else casting.where
        sentences.append(f"The person {who} becomes the person in <Picture {first}>.")
        sentences += _views(casting.roles, first)
    sentences.append(_GROUP_KEEP)
    for first, casting in zip(firsts, castings, strict=True):
        if casting.description.strip():
            sentences.append(
                f"The person in <Picture {first}>: {casting.description.strip()}"
            )
        if casting.unlike:
            sentences.append(
                f"Unlike the original person {casting.where}, they have "
                f"{_none_of(casting.unlike)}."
            )
    if notes.strip():
        sentences.append(_tagged(notes.strip(), firsts))
    return " ".join(sentences)


def _views(roles: Sequence[str], first: int) -> list[str]:
    """Says what each picture after the first shows, and the first if it's a back."""
    return [
        _ROLES.get(role, _OTHER_VIEW).format(n=n)
        for n, role in enumerate(roles, start=first)
        if n > first or role == "back"
    ]


def _none_of(items: Sequence[str]) -> str:
    *rest, last = [f"no {item}" for item in items]
    return f"{', '.join(rest)} and {last}" if rest else last


def _tagged(notes: str, firsts: Sequence[int]) -> str:
    """Replaces "@n" with the n-th person's character, by their first picture."""

    def name(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if not 0 <= index < len(firsts):
            return match.group(0)
        return f"the person in <Picture {firsts[index]}>"

    return _TAG.sub(name, notes)


def fill_gaps(masks: np.ndarray) -> np.ndarray:
    """Gives frames where the person was lost the mask of the nearest frame.

    H3 Relay refuses masks with empty frames, and a short gap is usually the
    person passing behind something.

    Args:
        masks: (frames, height, width) bools.

    Returns:
        A copy with every frame filled.

    Raises:
        ValueError: Every frame is empty.
    """
    seen = np.flatnonzero(masks.reshape(len(masks), -1).any(axis=1))
    if not len(seen):
        raise ValueError("The person wasn't found in any frame. Tap them again.")
    frames = np.arange(len(masks))
    nearest = seen[np.abs(frames[:, None] - seen[None, :]).argmin(axis=1)]
    return masks[nearest]


def paste(
    take: Path,
    footage: Path,
    masks: np.ndarray,
    dest: Path,
    pad: int,
    feather: float,
) -> None:
    """Writes the footage with only what H3 redrew taken from its take.

    H3 renders the whole frame at about a megapixel. Pasting back only the
    region it redrew keeps everything else as the footage's own pixels, at the
    footage's full size.

    Args:
        take: H3's render.
        footage: The same frames at full size.
        masks: (frames, height, width) bools in the footage's shape, at any
            size: the people H3 was told to redraw.
        dest: Where the result goes, silent.
        pad: How far past the masks H3 may have drawn, in its own pixels: the
            margin they were grown by, plus H3 Relay's cells.
        feather: The soft edge where the take meets the footage, in H3's pixels.
    """
    size = video_prep.probe(footage)
    canvas = video_prep.probe(take)
    scale = max(canvas.width / size.width, canvas.height / size.height)
    # The take shows the footage's middle: H3 Relay scaled it to cover its
    # canvas and cropped the rest.
    width, height = round(canvas.width / scale), round(canvas.height / scale)
    left, top = (size.width - width) // 2, (size.height - height) // 2

    def pasted() -> Iterator[np.ndarray]:
        frames = zip(
            video_prep.iter_frames(take), video_prep.iter_frames(footage), strict=True
        )
        for index, (redrawn, frame) in enumerate(frames):
            near = masks[max(index - PASTE_SPAN, 0) : index + PASTE_SPAN + 1]
            region = _cover(
                near.any(axis=0).astype(np.float32), canvas.width, canvas.height
            )
            # Everything within pad of the region: a round dilation, ~100x faster
            # than cv2.dilate with a kernel this size (0.9 s a frame at 768p).
            outside = (region <= 0.5).astype(np.uint8)
            distance = cv2.distanceTransform(
                outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            region = (distance <= pad).astype(np.float32)
            alpha = cv2.GaussianBlur(region, (0, 0), feather)
            alpha = cv2.resize(alpha, (width, height))[..., None]
            person = cv2.resize(
                redrawn, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
            middle = frame[top : top + height, left : left + width]
            middle[:] = (middle * (1 - alpha) + person * alpha).round().clip(0, 255)
            yield frame

    video_prep.write_frames(pasted(), FPS, (size.width, size.height), dest)


def _cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scales an image to cover width x height and crops the middle, like H3 Relay."""
    scale = max(width / image.shape[1], height / image.shape[0])
    scaled = cv2.resize(
        image, (round(image.shape[1] * scale), round(image.shape[0] * scale))
    )
    top, left = (scaled.shape[0] - height) // 2, (scaled.shape[1] - width) // 2
    return scaled[top : top + height, left : left + width]


def swap_graph(
    source: str | Asset,
    masks: str | Asset,
    pictures: Sequence[str | Asset],
    *,
    prompt: str,
    seed: int,
    resolution: str = "768",
) -> Graph:
    """Returns the workflow that swaps the masked person for the character.

    Args:
        source: The clip, silent, at 24 fps.
        masks: The person's masks as a video, white on black, one frame per
            frame of the clip, at any size with the clip's shape.
        pictures: The character's pictures, <Picture 1> first.
        prompt: What H3 is asked to do, from prompt().
        seed: The sampler's noise seed.
        resolution: One of RESOLUTIONS.

    Returns:
        The workflow; the take comes from the node "save".

    Raises:
        ValueError: There are no pictures or more than MAX_PICTURES, or the
            resolution is unknown.
    """
    if not 1 <= len(pictures) <= MAX_PICTURES:
        raise ValueError(f"Give H3 between 1 and {MAX_PICTURES} pictures.")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"The resolution is one of {', '.join(RESOLUTIONS)}.")
    references = {
        f"ref_images.ref_image_{index}": [f"picture_{index}", 0]
        for index in range(len(pictures))
    }
    return {
        "model": _node("UNETLoader", unet_name=MODEL, weight_dtype="default"),
        "shift": _node(
            "MiniMaxH3SigmaShift", model=["model", 0], shift_video=12.0, shift_audio=3.0
        ),
        "attention": _node(
            "ModelAttentionBackend",
            model=["shift", 0],
            attention="comfy kitchen attention",
        ),
        "turbo": _node(
            "LoraLoaderModelOnly",
            model=["attention", 0],
            lora_name=TURBO_LORA,
            strength_model=1.0,
        ),
        "swap_lora": _node(
            "LoraLoaderModelOnly",
            model=["turbo", 0],
            lora_name=SWAP_LORA,
            strength_model=1.0,
        ),
        "text": _node(
            "CLIPLoader", clip_name=TEXT_ENCODER, type="minimax", device="default"
        ),
        "text_device": _node("SelectCLIPDevice", clip=["text", 0], device="gpu:0"),
        "text_memory": _node("H3RelayTextMemory", clip=["text_device", 0]),
        "video_vae": _node("VAELoader", vae_name=VIDEO_VAE),
        "audio_vae": _node("VAELoader", vae_name=AUDIO_VAE),
        "source": _node("LoadVideo", file=source),
        "prepare": _node(
            "H3RelayPrepareSource",
            video=["source", 0],
            resolution=resolution,
            duration_seconds=0.0,
            missing_audio="silence",
        ),
        "dimensions": _node("GetImageSize", image=["prepare", 0]),
        **{
            f"picture_{index}": _node("LoadImage", image=picture)
            for index, picture in enumerate(pictures)
        },
        "mask_file": _node("LoadVideo", file=masks),
        "mask_frames": _node("GetVideoComponents", video=["mask_file", 0]),
        # H3 Relay scales the clip to cover its canvas and crops the middle;
        # the masks are fitted the same way so they stay on the person.
        "mask_fitted": _node(
            "ImageScale",
            image=["mask_frames", 0],
            upscale_method="nearest-exact",
            width=["dimensions", 0],
            height=["dimensions", 1],
            crop="center",
        ),
        "mask": _node("ImageToMask", image=["mask_fitted", 0], channel="red"),
        "expand_mask": _node(
            "H3RelayExpandMask",
            masks=["mask", 0],
            images=["prepare", 0],
            source_frames=["prepare", 5],
            expansion_px=round(EXPAND * _scale(resolution)),
        ),
        "plan": _node(
            "H3RelayWindowPlan",
            source_frames=["prepare", 5],
            window_frames=str(WINDOW_FRAMES),
            history_frames=str(HISTORY_FRAMES),
            seed=seed,
            schedule="adaptive",
            seed_mode="fixed",
            seed_overrides="{}",
        ),
        "start": _node(
            "StartLoop",
            mode="simple",
            cache_iterations=False,
            **{"mode.num_iterations": ["plan", 2]},
        ),
        "window": _node(
            "H3RelayWindowInputs",
            source=["prepare", 0],
            window=["plan", 3],
            iteration_index=["start", 0],
            previous=["start", 4],
            reference_history=True,
            fps=FPS,
            resize_anchor=True,
            history_audio_mode="source",
            source_audio=["prepare", 1],
            mask=["expand_mask", 0],
        ),
        "conditioning": _node(
            "MiniMaxH3ReferenceToVideo",
            clip=["text_memory", 0],
            prompt=prompt,
            width=["dimensions", 0],
            height=["dimensions", 1],
            length=["window", 8],
            ref_image_size="match",
            vae=["video_vae", 0],
            audio_vae=["audio_vae", 0],
            **references,
            **{
                "ref_videos.ref_video_0": ["window", 5],
                "ref_audios.ref_audio_0": ["window", 6],
            },
        ),
        "history": _node(
            "H3RelayApplyHistory",
            positive=["conditioning", 0],
            latent=["conditioning", 1],
            vae=["video_vae", 0],
            audio_vae=["audio_vae", 0],
            frames=["window", 3],
            history_frames=["window", 7],
            first_frame_guide=False,
            target_audio_guide=False,
            audio=["window", 4],
        ),
        "encode_source": _node("VAEEncode", pixels=["window", 0], vae=["video_vae", 0]),
        "edit_mask": _node(
            "H3RelayApplyEditMask",
            latent=["history", 1],
            source_latent=["encode_source", 0],
            mask=["window", 1],
        ),
        "guide": _node(
            "MiniMaxH3AddGuide",
            positive=["history", 0],
            latent=["edit_mask", 0],
            frame_idx=0,
            audio_vae=["audio_vae", 0],
            audio=["window", 2],
        ),
        "sampling_memory": _node(
            "H3RelaySamplingMemory",
            model=["swap_lora", 0],
            latent=["edit_mask", 0],
            conditioning=["guide", 0],
            activation_memory_gib=17.0,
        ),
        "noise": _node("RandomNoise", noise_seed=["window", 10]),
        "guider": _node(
            "CFGGuider",
            model=["sampling_memory", 0],
            positive=["guide", 0],
            negative=["guide", 0],
            cfg=1.0,
        ),
        "sampler": _node("KSamplerSelect", sampler_name="er_sde"),
        "schedule": _node(
            "BasicScheduler",
            model=["swap_lora", 0],
            scheduler="normal",
            steps=STEPS,
            denoise=1.0,
        ),
        SAMPLER: _node(
            "SamplerCustomAdvanced",
            noise=["noise", 0],
            guider=["guider", 0],
            sampler=["sampler", 0],
            sigmas=["schedule", 0],
            latent_image=["edit_mask", 0],
        ),
        "separate": _node("LTXVSeparateAVLatent", av_latent=[SAMPLER, 0]),
        "decode": _node("VAEDecode", samples=["separate", 0], vae=["video_vae", 0]),
        "append": _node(
            "H3RelayAppendWindow",
            frames=["decode", 0],
            window=["window", 12],
            previous=["start", 4],
        ),
        "end": _node(
            "EndLoop",
            output_value=["append", 0],
            next_iteration_value=["append", 0],
            accumulate=False,
        ),
        "finish": _node(
            "H3RelayFinishWindows",
            state=["end", 0],
            audio_mode="source",
            source_audio=["prepare", 1],
        ),
        "video": _node(
            "CreateVideo", images=["finish", 0], fps=["finish", 2], audio=["finish", 1]
        ),
        "save": _node(
            "SaveVideo",
            video=["video", 0],
            filename_prefix="rigless/swap",
            format="mp4",
            **{
                "format.codec": "h264",
                "format.codec.encoding": "re-encode",
                "format.codec.encoding.crf": 16,
            },
        ),
    }


def render_takes(
    part: Path,
    masks: Sequence[Path],
    pictures: Sequence[Path],
    *,
    prompt: str,
    seeds: Sequence[int],
    out_dir: Path,
    sound: tuple[Path, float] | None = None,
    full: Path | None = None,
    resolution: str = "768",
    on_progress: Callable[[int, Progress], None] | None = None,
) -> Iterator[Take]:
    """Renders and scores one swap take per seed, yielding each when it's done.

    Each take's folder holds h3.mp4 (H3's render, with the lead-in), result.mp4
    (the same frames as part, scored), finished.mp4
    (pasted onto the full-size part if given, with the clip's sound: the one
    to show), and run.json. A take that fails, in ComfyUI or while it's being
    finished, is yielded with its error, and the next seed goes on. With
    several people, a take scores as well as its least faithful person.

    Args:
        part: The clip to swap in, silent, at 24 fps.
        masks: Each person's masks as a video, one frame per frame of part.
        pictures: The characters' pictures, <Picture 1> first.
        prompt: What H3 is asked to do, from prompt().
        seeds: The seeds to render, one take each, in order.
        out_dir: Where the take folders go.
        sound: The clip part came from and where part starts in it, in
            seconds, for each take's soundtrack; None leaves takes silent.
        full: The same frames as part at full size, to paste each take onto;
            None shows H3's render as it is.
        resolution: One of RESOLUTIONS.
        on_progress: Called with the take's index in seeds and each progress
            update while it renders.

    Yields:
        Each take, in seed order.

    Raises:
        ValueError: seeds contains a duplicate, or a person was never found.
    """
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each seed may appear only once.")
    out_dir.mkdir(parents=True, exist_ok=True)
    people = [fill_gaps(video_prep.read_frames(path)[..., 0] > 127) for path in masks]
    found = reduce(np.logical_or, people)
    boxes = [pose.mask_boxes(person) for person in people]
    dancers = [pose.estimate_in(part, person_boxes) for person_boxes in boxes]
    frames = len(found)
    lead = min(LEAD_IN, frames - 1)
    led = video_prep.lead_in_order(frames, lead)  # the masks in part.mp4's order
    video_prep.write_video(
        found[led].astype(np.uint8) * 255, FPS, out_dir / "masks.mp4"
    )
    video_prep.lead_in(part, lead, FPS, out_dir / "part.mp4")
    with comfy_client.connect() as client:
        part_asset = client.assets.from_file(out_dir / "part.mp4")
        masks_asset = client.assets.from_file(out_dir / "masks.mp4")
        picture_assets = [client.assets.from_file(path) for path in pictures]
        for index, seed in enumerate(seeds):
            take_dir = out_dir / f"seed-{seed}"
            take_dir.mkdir()
            graph = swap_graph(
                part_asset,
                masks_asset,
                picture_assets,
                prompt=prompt,
                seed=seed,
                resolution=resolution,
            )
            progress = partial(on_progress, index) if on_progress else None
            started = time.monotonic()
            job_id, error = comfy_client.try_run(
                client, graph, {"save": take_dir / "h3.mp4"}, progress
            )
            render_seconds = round(time.monotonic() - started, 1)
            take_score = None
            if not error:
                try:
                    result = take_dir / "result.mp4"
                    video_prep.cut_frames(
                        take_dir / "h3.mp4", result, lead, lead + frames
                    )
                    take_score = _score(
                        result, dancers, boxes, (found.shape[2], found.shape[1])
                    )
                    shown = result
                    if full:
                        shown = take_dir / "pasted.mp4"
                        scale = _scale(resolution)
                        pad = round((PASTE_PAD + EXPAND) * scale)
                        paste(result, full, found, shown, pad, FEATHER * scale)
                    # Without a soundtrack, the take's own (silent) one goes back on.
                    source, start = sound or (result, 0.0)
                    video_prep.add_audio(
                        shown,
                        source,
                        start,
                        take_dir / "finished.mp4",
                        comment=AI_LABEL,
                    )
                # ffmpeg, the pose model or a render with the wrong frame count.
                except (RuntimeError, ValueError, OSError) as exc:
                    take_score = None
                    error = f"The take rendered, but finishing it failed: {exc}"
            record: dict[str, Any] = {
                "model": MODEL,
                "swap_lora": SWAP_LORA,
                "resolution": resolution,
                "prompt": prompt,
                "pictures": [path.name for path in pictures],
                "seed": seed,
                "status": "failed" if error else "done",
                "error": error,
                "job_id": job_id,
                "render_seconds": render_seconds,
                "score": take_score.as_record() if take_score else None,
            }
            (take_dir / "run.json").write_text(json.dumps(record, indent=2), "utf-8")
            yield Take(seed, take_dir, error, take_score)


def _score(
    result: Path,
    dancers: Sequence[pose.PoseTrack],
    boxes: Sequence[Sequence[np.ndarray | None]],
    size: tuple[int, int],
) -> TakeScore:
    """Scores each replaced person in a take and returns the least faithful one's.

    Args:
        result: H3's render.
        dancers: Each person's poses in the clip.
        boxes: Each person's boxes in the clip, from pose.mask_boxes.
        size: The clip's (width, height).
    """
    info = video_prep.probe(result)
    canvas = (info.width, info.height)
    scores = [
        score_take(
            dancer, pose.estimate_in(result, pose.fit_boxes(found, size, canvas))
        )
        for dancer, found in zip(dancers, boxes, strict=True)
    ]
    return min(scores, key=_rank)


def _scale(resolution: str) -> float:
    """Returns how lengths in H3's pixels at 768p scale to one of RESOLUTIONS."""
    return int(resolution) / 768


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}
