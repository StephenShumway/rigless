<p align="center"><img src="src/rigless/web/logo.svg" width="96" alt="Rigless logo"></p>

<h1 align="center">Rigless</h1>

<p align="center"><b>Swap anyone in a clip for your character. No rig, no suit, no prompt.</b></p>

<!-- TODO before release: a GIF of a finished swap here (examples/). -->

Rigless is an open-source app that redraws one or more people in real footage as the
character in your pictures. You click the person; everything else (the scene, the
camera, the light, everyone else in the shot) stays exactly as filmed. It runs on open
weights only, with [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) as the core
model, through the [Comfy SDK](https://pypi.org/project/comfy-sdk/), on a GPU you choose.

Built for Comfy's [#OpenSourceHiggsfield](https://comfy.org/open-source-higgsfield)
challenge, as an open answer to Higgsfield's Genjutsu.

## What makes it different

- **The scene stays pixel-exact.** H3 redraws only inside the tracked person's mask, and
  the take is pasted back onto your full-size footage. Outside the person you see the
  frames you filmed, at their own resolution, not a re-drawn copy.
- **Its own LoRA, trained on real footage.** The Rigless LoRA fine-tunes H3 Relay's
  Character Swap LoRA so it reads your character's pictures closely and matches the
  footage's light. See [The Rigless LoRA](#the-rigless-lora).
- **No prompt needed.** An open vision model (Qwen3-VL 8B) describes your character from
  their pictures and lists what the performer has that the character doesn't ("no rings,
  no watch"), so H3 doesn't carry those over. You can add your own notes.
- **Every take is measured.** Each take gets a motion score from 0 to 100 against the
  performance, frozen and mirrored takes are flagged, and the best take is marked for you.
- **Click, don't mask.** Click the person and SAM 3.1 tracks them through the clip, with a
  preview that shows where it lost them. Add up to three people, each with their own
  character; one pass swaps them all.
- **Bring your own GPU.** Deploy a RunPod GPU from inside the app with your own key, use
  your own NVIDIA GPU, or point it at a ComfyUI anywhere else.

## Using it

1. **Drop a clip** (up to 2 minutes) and trim it to the part you want, up to 20 seconds.
2. **Click the person** to replace. Check the tracking preview and tap again where it lost
   them, for example after a camera cut: a tap on another frame fixes the tracking from
   that frame on. To pick someone else, tap them on the same frame, or **Clear taps** on
   the person's card. **+ Add person** swaps more than one.
3. **Add the character:** a character sheet, or a few pictures: a front view, a close-up
   of the face and a back view work best. Small details such as a monocle or a ring need
   the close-up; from one front photo they can come out wrong.
4. **Generate** a **Draft** (480p, about 2.5x faster, for checking casting and motion) or a
   **Full** take (768p). Takes add up; compare them, and download the best.

The first render asks you to agree to MiniMax H3's terms (see
[Responsible use](#responsible-use)).

## Install

You need [uv](https://docs.astral.sh/uv/) and git. The app runs on Windows, macOS or Linux;
the rendering runs on a GPU with **48 GB or more** (tested on an A40 48 GB and an H100
80 GB) and a driver for **CUDA 13**.

```bash
git clone https://github.com/StephenShumway/rigless
cd rigless
uv sync
uv run rigless
```

The app opens in your browser at `http://127.0.0.1:8190`. It only listens on your own
computer. Then choose where it renders with the button at the top right:

### RunPod (easiest)

1. Make a [RunPod](https://www.runpod.io/) account, add credit, and create an API key.
2. In the app, choose **RunPod**, paste the key (it stays in your browser), pick a GPU, and
   press **Deploy**. Rigless rents a GPU in your account and sets it up: ComfyUI, the models
   (about 67 GB) and a token-protected connection. The first setup takes about 15 minutes, and the app shows its progress.
3. **Pause** when you're done: RunPod stops billing the GPU and keeps the disk (about $0.03
   an hour for its 100 GB), and **Resume** brings it back in a few minutes. **Delete** removes
   both.

### This Computer

For an NVIDIA GPU with 48 GB or more in your own machine:

1. Install [ComfyUI](https://github.com/Comfy-Org/ComfyUI) **v0.41.2** and the
   [H3 Relay](https://github.com/akatz-ai/h3-relay) nodes in its `custom_nodes` (the commit
   Rigless was tested with is pinned in `deploy/runpod/setup.sh`).
2. Download the models into ComfyUI's models folder:
   `uv run python scripts/download_models.py --models-dir /path/to/ComfyUI/models`
3. Start ComfyUI, then the proxy that Rigless talks to:
   `uv run comfy-api-proxy start --comfyui http://127.0.0.1:8188` (it listens on
   `127.0.0.1:8189`).
4. In the app, choose **This Computer**.

`deploy/runpod/setup.sh` does all of this on Linux and is the exact recipe.

### Custom (any other GPU server)

On any Linux machine with a suitable GPU (another cloud, a lab server), clone this repo and
run `PROXY_TOKEN=<a long random value> bash deploy/runpod/setup.sh`. It installs everything
under `/workspace` and starts the proxy on port 8189, protected by that token. Expose the
port over HTTPS. Then copy `.env.example` to `.env` in your local copy of Rigless and fill
in two lines:

```bash
COMFY_BASE_URL=https://your-server:8189  # where the proxy is
COMFY_API_KEY=<the same token>           # what the proxy checks on every request
```

These are the Comfy SDK's standard names. `COMFY_API_KEY` is only needed because the proxy
has a token; a proxy started without one (only sensible on a private network) needs just
`COMFY_BASE_URL`. Restart Rigless and it renders there, shown as **Custom**. Remove the
lines to choose RunPod or This Computer again.

## How it works

1. **Footage.** The trimmed part is converted to a steady 24 fps, H3's frame rate, so
   every step counts the same frames.
2. **Tracking.** Your clicks go to SAM 3.1 in ComfyUI, which tracks each person through the
   part. If a mask covers only part of a body, Rigless adds taps on the body and tracks
   again.
3. **Describing.** Qwen3-VL 8B describes the character's appearance from all their pictures
   (never their pose or expression, which H3 would act out), compares them with the
   performer to list what the character doesn't have, and checks the pictures and
   footage are safe for work.
4. **Swapping.** MiniMax H3 redraws each person inside their mask, grown by a margin for
   hair and clothes, in overlapping windows, with the clip as `<Video 1>` and the pictures
   as `<Picture 1>` to `<Picture N>`. Outside the masks H3 keeps the clip's own latents.
   It runs with an 8-step Turbo LoRA and the Rigless LoRA, and hears the clip silent, so it
   follows the performer's mouth rather than the soundtrack.
5. **Pasting.** The take is pasted onto the full-size footage (up to 1920 px) through a
   soft-edged mask, and gets the clip's own sound back. Its metadata says it's
   AI-generated.
6. **Scoring.** RTMPose finds the performer's pose in the clip and the character's pose in
   the take, inside the same boxes, and compares limb directions frame by frame. A take
   that barely moves is flagged frozen, one that moves like a mirror image is flagged
   mirrored. The best take is the best unflagged one, Full before Draft.

## Models and tools

| What | Used for | License |
|---|---|---|
| [MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (ref2va pruned int8, video and audio VAEs) | The swap | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) |
| Qwen3-VL-32B, H3's text encoder (int8, in the same repo) | Reading H3's prompt | Apache-2.0 (per the MiniMax H3 license's note) |
| [H3 Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo) (FL2V 8-step, 768p) | Sampling in 8 steps | MiniMax H3 Community License, as an H3 derivative |
| [Rigless LoRA](https://huggingface.co/StephenShumway/rigless-lora) | The swap, fine-tuned on real footage | MiniMax H3 Community License |
| [Character Swap LoRA v1.5](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA) by akatz | What the Rigless LoRA was fine-tuned from | MiniMax H3 Community License |
| [H3 Relay](https://github.com/akatz-ai/h3-relay) nodes by akatz | Masked, windowed H3 rendering | GPL-3.0, installed separately in ComfyUI |
| [SAM 3.1](https://huggingface.co/Comfy-Org/sam3.1) | Tracking the people you click | [SAM License](https://huggingface.co/Comfy-Org/sam3.1/blob/main/LICENSE) |
| Qwen3-VL 8B ([Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1), `text_encoders`) | The description, the "has no" list, the safety check | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE): research and evaluation only |
| [rtmlib](https://github.com/Tau-J/rtmlib) with RTMPose and YOLOX | Poses for scoring and the tracking checks, views on sheets | Apache-2.0; the weights come from OpenMMLab <!-- TODO: confirm the RTMPose/YOLOX weights' license --> |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Running the models | GPL-3.0, a separate program |
| [Comfy SDK](https://github.com/Comfy-Org/comfy-python-sdk), [comfy-api-proxy](https://github.com/Comfy-Org/comfy-api-proxy) | Talking to ComfyUI | MIT |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | Video in and out (it bundles an FFmpeg build) | BSD-2-Clause |
| [IBM Plex](https://github.com/IBM/plex) Sans and Mono | The app's fonts, bundled | SIL Open Font License 1.1 (`src/rigless/web/fonts/OFL.txt`) |

The Qwen3-VL 8B file is under Qwen's research license, unlike the Apache-2.0 Qwen3-VL
release, so the describer as shipped is for non-commercial use.

## The Rigless LoRA

`rigless-lora-v1.safetensors` on Hugging Face:
[StephenShumway/rigless-lora](https://huggingface.co/StephenShumway/rigless-lora).
Rigless downloads it with the other models.

It fine-tunes akatz's H3 Relay Character Swap LoRA v1.5 on real footage. Each training
sample is a real clip with someone else swapped in by H3; the LoRA learns to turn that
back into the real person, given only a picture of them. The target is always real
footage, so it learns real light and has to read the picture to know who to draw.
<!-- TODO: the clip count and sources (46 clips: 30 Pexels + 16 others?) and the Pexels credit. -->

In a blind test it was preferred over v1.5 and over no swap LoRA (3 clips, one take each,
judged by its developer). It is not measurably better at face identity.

### Compare with v1.5

Download `h3_character_swap_v1.5.safetensors` from
[akatz-ai/MiniMax-H3-Character-Swap-LoRA](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA)
into ComfyUI's `models/loras`, and set `SWAP_LORA` in `src/rigless/swap.py` to that file
name.

## Limitations

- One take covers up to 20 seconds of footage, and replaces up to three people.
- The GPU needs 48 GB or more; smaller cards are untested.
- Small accessories need a close-up of the face; from a single front photo they can come
  out wrong (a monocle drawn as glasses).
- After a camera cut, SAM can lose the person or follow someone else; check the preview and
  tap again.

## Responsible use

Rigless renders with MiniMax H3, licensed under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
Before the first render the app shows its terms and asks you to agree. Among other things:
use only footage and pictures you have the rights to, of people who consented or who
aren't real; never impersonate, harass, deceive or harm anyone; say clearly that a video
is AI-generated wherever you share it; and never use H3's output to train another AI
model. MiniMax H3 can't be used in the United States, the European Union, the United
Kingdom or South Korea without your own
[authorization from MiniMax](https://platform.minimax.io/h3-license).

Report misuse: <!-- TODO: the report email (terms.REPORT_EMAIL) -->

## For developers

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run mypy
uv run pytest
```

Tests that need a running ComfyUI carry the `comfy` marker, and tests that need the local
test videos carry `fixtures`; both skip cleanly when those are missing.

Settings, all optional, go in `.env` at the repo root (see `.env.example`):

| Variable | What it does |
|---|---|
| `COMFY_BASE_URL` | A ComfyUI anywhere else: its comfy-api-proxy address. Overrides the choice in the app. |
| `COMFY_API_KEY` | That proxy's bearer token, if it has one. |

| Module | Role |
|---|---|
| `app.py` | The web app and its API |
| `projects.py` | Shoots: footage, people, pictures, renders and takes |
| `selection.py` | Tracking the clicked people with SAM 3.1 |
| `sheet.py` | Finding the views on a character sheet |
| `describe.py` | Qwen3-VL: the description, the "has no" list, the safety check |
| `swap.py` | The H3 swap graph, rendering, pasting |
| `pose.py`, `score.py` | Pose estimation and the motion score |
| `video_prep.py` | ffmpeg and OpenCV helpers |
| `comfy_client.py`, `config.py` | The connection to ComfyUI |
| `runpod.py` | Deploying and managing the RunPod GPU |
| `terms.py` | The MiniMax H3 terms each user agrees to |

## License

The code is under the [Apache License 2.0](LICENSE). The models are under their own
licenses (see [Models and tools](#models-and-tools)); the Rigless LoRA, a derivative of
MiniMax H3, is under the MiniMax H3 Community License Agreement.
