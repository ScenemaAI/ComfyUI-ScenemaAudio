# ComfyUI-ScenemaAudio

Native ComfyUI nodes for [Scenema Audio](https://scenema.ai/audio) — expressive text-to-speech with zero-shot voice cloning, built on the LTX 2.3 audio diffusion transformer.

Not just TTS: real vocal performance. Laughs, whispers, voice cracks, breath, singing, foreign accents — all driven by prompt cues.

## Quick start

1. Install (see below).
2. Set up a HuggingFace token (see below).
3. Load `workflows/scenema_audio.json` in ComfyUI, pick a preset from the dropdown, hit Queue.

## Installation

### ComfyUI Manager (recommended)

- Manager tab → **Install via Git URL**
- Paste `https://github.com/ScenemaAI/ComfyUI-ScenemaAudio.git`
- Restart ComfyUI

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ScenemaAI/ComfyUI-ScenemaAudio.git
pip install -r ComfyUI-ScenemaAudio/requirements.txt
```

Restart ComfyUI. All model weights auto-download from HuggingFace on first use (~30 GB total, one-time).

## HuggingFace token (required)

The text encoder is [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it), which is a **gated model**. You must:

1. Visit https://huggingface.co/google/gemma-3-12b-it and click **Agree and access repository**.
2. Create a token at https://huggingface.co/settings/tokens (read scope is enough).
3. Make the token available to ComfyUI in one of two ways:

   ```bash
   # Option A — persistent, recommended
   huggingface-cli login
   # paste your token when prompted

   # Option B — environment variable (per-session)
   export HF_TOKEN=hf_...
   ```

If you skip this step, the first generation will fail with clear instructions. No cryptic 401s.

## Hardware requirements

**Minimum: 8 GB VRAM.** Tested on RTX 3070 (8 GB) and RTX 4090 (24 GB).

| GPU tier | Gemma path (auto) | Peak VRAM | Speed (33 s output) |
|---|---|---|---|
| **8 GB** (e.g. RTX 3070) | bf16 streams from CPU RAM | ~6.7 GB | ~2 min (3-4× realtime) |
| **12 GB+** (e.g. RTX 3060 12 GB) | NF4 quantized on GPU | ~15 GB | ~1 min (2× realtime) |
| **24 GB** (e.g. RTX 4090) | NF4 quantized on GPU | ~21 GB | ~40 s (1.2× realtime), ~15 s from warm cache |

Also needs **~32 GB system RAM** for the pipeline components. Nothing is on-GPU when you're not generating.

## The 12 nodes

| Node | Purpose |
|---|---|
| **Scenema Audio Generate** | Main node. Voice description + text + scene → expressive audio. Includes preset dropdown with 12 curated voices from the [scenema.ai/audio](https://scenema.ai/audio) demos. |
| **Scenema Audio Model Loader** | Loads the 3.3 B audio transformer (INT8 by default; bf16 available if you have 12 GB+). |
| **Scenema Audio VAE Loader** | Loads the audio VAE encoder + decoder. |
| **Scenema Audio VAE Encode** | Encodes reference audio to a latent for voice cloning. Capped at 20 s. |
| **Scenema Audio Load Audio URL** | Loads a reference audio file from a URL (mp3/wav/flac) for voice cloning. Alternative to the built-in LoadAudio. |
| **Scenema Audio Voice Clone** | Standalone SeedVC voice conversion (source → reference voice). Also invoked automatically by Generate when a reference is provided. |
| **Scenema Audio Text Encode** | Standalone text encoder (Gemma 3 12B). Rarely needed directly — Generate uses it internally. |
| **Scenema Audio Sampler** | Standalone 8-step diffusion sampler. Advanced use only. |
| **Scenema Audio Decode** | Standalone audio latent decoder. |
| **Scenema Audio Chunker** | Kokoro-based text chunker for very long text. |
| **Scenema Audio Concatenate** | Joins multiple audio outputs. |

## Writing prompts

> **Note on XML:** raw `<speak>`/`<action>` XML input is no longer supported in the ComfyUI nodes. Everything is entered as plain fields — voice description, speech text, action tags. The XML the model expects is built for you internally. If you're coming from the original Scenema Audio API docs, use the field-based syntax below.

Three fields drive expressiveness:

**Voice description** — describes the speaker (age, gender presentation, timbre, accent, delivery):
```
Male, mid 50s. Refined Central European accent with an Austrian tinge.
Warm baritone that turns cold in an instant. Cultured, articulate,
dangerously calm.
```

**Action tags** — opening delivery cues, one per line. Sets the overall performance for the speech:
```
He smiles as he speaks, without warmth
```

**Speech text** — the words to say. Use inline `[bracketed cues]` for **mid-speech** performance direction. Each bracket becomes a stage direction the model performs at that exact position in the text:

```
You know, [he lets out a soft, dry laugh] I've always found politeness
to be such a charming way of holding a knife. [His voice drops, suddenly
intimate] And all the while, their hands are already reaching for you.
```

Available bracket cues: laughs, whispers, voice cracks, gasps, sobs, sighs, pauses, mood shifts, emphasis, breath sounds — anything you'd write as a stage direction in a script. The model was trained on film-style narration and responds to natural language cues.

**Where to put what:**
- Opening tone / posture → `action_tags` field.
- Mid-speech emotional shifts, laughs, whispers → inline `[brackets]` in `speech_text`.

## Preset dropdown

Twelve production-tested voices from the [official demos](https://scenema.ai/audio), auto-fill all fields when selected:

- **Old Male Storyteller** (fireside, Southern American)
- **Young Woman** (breathless discovery)
- **Terrified Whisper** (hiding)
- **Irish Woman, Dry Wit**
- **Rage to Vulnerability** (Italian-American, arc)
- **Eulogy** (grief, extremely slow)
- **Terror** (sobbing, hyperventilating)
- **Villain** (laughing menace)
- **Rain and Thunder** (SFX + urgent shouting)
- **Italian Cooking Show** (SFX + enthusiastic)
- **Kid Explaining Dinosaurs** (8 year old)
- **British Woman, East London Rage**

Custom lets you write your own from scratch.

## Voice cloning

Two paths for cloning from a real voice:
1. **Load Audio** (built-in ComfyUI node) → **Scenema Audio VAE Encode** → connect to Generate's `ref_latent` input.
2. **Scenema Audio Load Audio URL** (paste a link) → **VAE Encode** → `ref_latent`.

Reference clip is capped at 20 s (longer doesn't help quality). SeedVC runs automatically after generation to polish voice identity match.

## Languages

Twelve tested: English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Hindi, Arabic, Swahili. Pick from the dropdown; write the `speech_text` in the target language.

## Links

- [Demos + full article](https://scenema.ai/audio)
- [Model weights](https://huggingface.co/ScenemaAI/scenema-audio)
- [Standalone Docker server](https://github.com/ScenemaAI/scenema-audio)

## License

Node code: MIT (see [LICENSE](LICENSE)).
Model weights: [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).
