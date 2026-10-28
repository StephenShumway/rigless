"""Words for the swap prompt, written from the pictures so the user needn't.

Without a description, H3 gets small or unusual details wrong (a monocle drawn
as round glasses) and carries over what the replaced person wears (their rings
onto the character's hands). Qwen3-VL-8B, the text encoder Qwen-Image 2.1 ships
with (under the Qwen Research License; see the README), looks at the
character's pictures side by side and describes them. It then answers yes or
no, item by item, for what the replaced person wears and the character
doesn't; it follows that format far more reliably than a request for a list.

Typical usage example:

  text = character(pictures, work_dir)
  unlike = differences(pictures, performer, work_dir)
  prompt = swap.prompt(roles, description=text, unlike=unlike)
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from comfy_sdk import Asset
from PIL import Image

from rigless import comfy_client
from rigless.comfy_client import Graph

VLM = "qwen3vl_8b_int8_convrot.safetensors"
COLLAGE_HEIGHT = 768
GAP = 16

DESCRIBE = (
    "These pictures show one person from several views. Describe their appearance "
    "so a video model can redraw them exactly: face, hair, facial hair, eyewear, "
    "jewelry and other accessories, and the clothing from top to bottom including "
    "shoes. Name colors and materials. Study any close-up of the face for small "
    "details such as a monocle (a single lens over one eye, often on a chain), "
    "earrings or piercings, and state them precisely. Describe only how they look, "
    "never their pose, gestures, expression or what they're doing: the video "
    "decides those. Don't say left or right. One paragraph under 90 words, no "
    "preamble."
)
# What H3 tends to carry over from the person it replaces: the name used in the
# prompt, and how the question puts it.
CARRY_OVER = {
    "glasses": "glasses (not sunglasses)",
    "sunglasses": "sunglasses",
    "rings": "rings on their fingers",
    "necklace": "a necklace or chain",
    "bracelet": "a bracelet",
    "watch": "a wristwatch",
    "earrings": "earrings",
    "hat": "a hat or cap",
    "headphones": "headphones",
    "tattoos": "visible tattoos",
    "dreadlocks": "dreadlocks",
    "braids": "braided hair",
    "long hair": "hair longer than their shoulders",
}
_QUESTION = (
    "Does the most prominent person in this picture have {}? Answer with only yes "
    "or no."
)
_LOOK = (
    "In at most eight words, what is the most prominent person in this picture "
    "wearing? Start with 'in', like 'in a red hoodie and blue jeans'. No sentence."
)
_LEAD = re.compile(r"^(in|wearing)\s+", re.IGNORECASE)
_UNSAFE = (
    "Does any part of this image show nudity, sexual content, or a child in a "
    "sexual or harmful situation? Answer with only yes or no."
)


def collage(images: Sequence[Image.Image]) -> Image.Image:
    """Puts images side by side at one height, on white, in order."""
    scaled = [
        image.convert("RGB").resize(
            (round(image.width * COLLAGE_HEIGHT / image.height), COLLAGE_HEIGHT)
        )
        for image in images
    ]
    width = sum(image.width for image in scaled) + GAP * (len(scaled) - 1)
    sheet = Image.new("RGB", (width, COLLAGE_HEIGHT), "white")
    x = 0
    for image in scaled:
        sheet.paste(image, (x, 0))
        x += image.width + GAP
    return sheet


def ask_graph(image: str | Asset, questions: Sequence[str], max_length: int) -> Graph:
    """Returns a workflow that asks several questions about one image.

    Args:
        image: The image.
        questions: What to ask, each on its own.
        max_length: The longest answer, in tokens.

    Returns:
        The workflow; question i's answer is saved as text by the node
        "save_<i>".
    """
    graph: Graph = {
        "vlm": _node(
            "CLIPLoader", clip_name=VLM, type="stable_diffusion", device="default"
        ),
        "image": _node("LoadImage", image=image),
    }
    for index, question in enumerate(questions):
        graph[f"ask_{index}"] = _node(
            "TextGenerate",
            clip=["vlm", 0],
            prompt=question,
            image=["image", 0],
            max_length=max_length,
            sampling_mode="off",
        )
        graph[f"save_{index}"] = _node(
            "SaveText",
            text=[f"ask_{index}", 0],
            filename_prefix="rigless/describe",
            format="txt",
        )
    return graph


def said_yes(answer: str) -> bool:
    """Returns whether a yes-or-no answer is yes."""
    return answer.strip().strip("*").lower().startswith("yes")


def character(pictures: Sequence[Path], work_dir: Path) -> str:
    """Describes the character from all their pictures at once.

    Args:
        pictures: The character's pictures.
        work_dir: Where the collage and answer are saved.

    Raises:
        comfy_sdk.ComfyError: ComfyUI couldn't run the workflow.
    """
    images = [Image.open(path) for path in pictures]
    (answer,) = _ask(images, [DESCRIBE], work_dir, max_length=256)
    return " ".join(answer.split())


def has(
    images: Sequence[Image.Image], items: Sequence[str], work_dir: Path
) -> list[str]:
    """Returns which of the items, keys of CARRY_OVER, the person shown has.

    Args:
        images: The person, in one or more pictures seen side by side.
        items: The keys of CARRY_OVER to ask about.
        work_dir: Where the collage and answers are saved.

    Raises:
        comfy_sdk.ComfyError: ComfyUI couldn't run the workflow.
    """
    questions = [_QUESTION.format(CARRY_OVER[item]) for item in items]
    answers = _ask(images, questions, work_dir, max_length=4)
    return [
        item for item, answer in zip(items, answers, strict=True) if said_yes(answer)
    ]


def unsafe(images: Sequence[Image.Image], work_dir: Path) -> bool:
    """Returns whether the images, seen side by side, show anything unsafe for work.

    One of the safeguards H3's license asks for: a yes stops the render.

    Args:
        images: The pictures and frames to check.
        work_dir: Where the collage and answer are saved.

    Raises:
        comfy_sdk.ComfyError: ComfyUI couldn't run the workflow.
    """
    (answer,) = _ask(images, [_UNSAFE], work_dir, max_length=4)
    return said_yes(answer)


def look(performer: Image.Image, work_dir: Path) -> str:
    """Says what a person in the clip wears, to tell them apart from others.

    Args:
        performer: The person, cropped from the footage.
        work_dir: Where the image and answer are saved.

    Returns:
        A short phrase such as "in a red hoodie and blue jeans".

    Raises:
        comfy_sdk.ComfyError: ComfyUI couldn't run the workflow.
    """
    (answer,) = _ask([performer], [_LOOK], work_dir, max_length=24)
    phrase = _LEAD.sub("", " ".join(answer.split())).strip(" .'\"")
    return f"in {phrase}" if phrase else ""


def differences(
    pictures: Sequence[Path], performer: Image.Image, work_dir: Path
) -> list[str]:
    """Returns what the replaced person has that the character doesn't.

    Args:
        pictures: The character's pictures.
        performer: The person being replaced, cropped from the footage.
        work_dir: Where the images and answers are saved.

    Returns:
        Keys of CARRY_OVER, in its order.

    Raises:
        comfy_sdk.ComfyError: ComfyUI couldn't run the workflow.
    """
    worn = has([performer], list(CARRY_OVER), work_dir)
    if not worn:
        return []
    theirs = set(has([Image.open(path) for path in pictures], worn, work_dir))
    return [item for item in worn if item not in theirs]


def _ask(
    images: Sequence[Image.Image],
    questions: Sequence[str],
    work_dir: Path,
    *,
    max_length: int,
) -> list[str]:
    """Shows the model the images side by side and returns its answers."""
    work_dir.mkdir(parents=True, exist_ok=True)
    asked = len(list(work_dir.glob("collage-*.png")))
    sheet = work_dir / f"collage-{asked}.png"
    collage(images).save(sheet)
    answers = {
        f"save_{index}": work_dir / f"answer-{asked}-{index}.txt"
        for index in range(len(questions))
    }
    with comfy_client.connect() as client:
        graph = ask_graph(client.assets.from_file(sheet), questions, max_length)
        comfy_client.download(comfy_client.run(client, graph), answers)
    return [path.read_text("utf-8").strip() for path in answers.values()]


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}
