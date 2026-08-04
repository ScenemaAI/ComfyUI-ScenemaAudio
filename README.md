# ComfyUI-ScenemaAudio

Native ComfyUI nodes for [Scenema Audio](https://scenema.ai/audio). Expressive text-to-speech with zero-shot voice cloning, built on the LTX 2.3 audio diffusion transformer.

This is not just TTS. It is real vocal performance. Laughs, whispers, voice cracks, breath, singing, foreign accents. All driven by prompt cues.

## Quick start

1. Install (see below).
2. Set up a HuggingFace token (see below).
3. Load `workflows/scenema_audio.json` in ComfyUI, pick a preset from the dropdown, hit Queue.

## Installation

### ComfyUI Manager (recommended)

Open the Manager tab. Choose **Install via Git URL**. Paste this URL:

```
https://github.com/ScenemaAI/ComfyUI-ScenemaAudio.git
```

Restart ComfyUI.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ScenemaAI/ComfyUI-ScenemaAudio.git
pip install -r ComfyUI-ScenemaAudio/requirements.txt
```

Restart ComfyUI. All model weights auto-download from HuggingFace on first use. Total download is around 30 GB, one time.

## HuggingFace token (required)

The text encoder is [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it), which is a gated model. You must:

1. Visit https://huggingface.co/google/gemma-3-12b-it and click **Agree and access repository**.
2. Create a token at https://huggingface.co/settings/tokens. Read scope is enough.
3. Make the token available to ComfyUI in one of two ways.

   ```bash
   # Option A. Persistent, recommended.
   huggingface-cli login
   # paste your token when prompted

   # Option B. Environment variable, per session.
   export HF_TOKEN=hf_...
   ```

If you skip this step the first generation fails with clear instructions. No cryptic 401s.

## Hardware requirements

Minimum is 8 GB VRAM. Tested end to end on RTX 3070 (8 GB) and RTX 4090 (24 GB).

| GPU tier | Gemma path (auto) | Peak VRAM | Speed (33 s output) |
|---|---|---|---|
| **8 GB** (e.g. RTX 3070) | bf16 streams from CPU RAM | 6.7 GB | around 2 minutes, 3 to 4x realtime |
| **12 GB and up** (e.g. RTX 3060 12 GB) | NF4 quantized on GPU | 15 GB | around 1 minute, 2x realtime |
| **24 GB** (e.g. RTX 4090) | NF4 quantized on GPU | 21 GB | around 40 seconds first pass, around 15 seconds from warm cache. 1 to 2x realtime. |

Also needs around 32 GB system RAM for the pipeline components. Nothing is on GPU when you are not generating.

## Nodes

Six user facing nodes.

| Node | Purpose |
|---|---|
| **Scenema Audio Generate** | The main node. Voice description plus text plus scene produces expressive audio. Includes a preset dropdown with 12 curated voices from the [scenema.ai/audio](https://scenema.ai/audio) demos. |
| **Scenema Audio Model Loader** | Loads the 3.3B audio transformer (INT8 by default, bf16 available if you have 12 GB or more). |
| **Scenema Audio VAE Loader** | Loads the audio VAE encoder and decoder. |
| **Scenema Audio VAE Encode** | Encodes reference audio to a latent for voice cloning. Capped at 20 seconds. |
| **Scenema Audio Load Audio from URL** | Loads a reference audio file from a URL (mp3, wav, flac) for voice cloning. Alternative to the built in LoadAudio. |
| **Scenema Audio Voice Clone** | Standalone SeedVC voice conversion (source to reference voice). Also invoked automatically by Generate when a reference is provided. |

## Writing prompts

> **Note on XML.** Raw `<speak>` and `<action>` XML input is no longer supported in the ComfyUI nodes. Everything is entered as plain fields. Voice description, action tags, and speech text. The XML the model expects is built for you internally. If you are coming from the original Scenema Audio API docs, use the field based syntax below.

Three fields drive expressiveness.

**Voice description.** Describes the speaker. Age, gender presentation, timbre, accent, delivery.

```
Male, mid 50s. Refined Central European accent with an Austrian tinge.
Warm baritone that turns cold in an instant. Cultured, articulate,
dangerously calm.
```

**Action tags.** Opening delivery cues, one per line. Sets the overall performance for the speech.

```
He smiles as he speaks, without warmth
```

**Speech text.** The words to say. Use inline `[bracketed cues]` for mid speech performance direction. Each bracket becomes a stage direction the model performs at that exact position in the text.

```
You know, [he lets out a soft, dry laugh] I've always found politeness
to be such a charming way of holding a knife. [His voice drops, suddenly
intimate] And all the while, their hands are already reaching for you.
```

Available bracket cues include laughs, whispers, voice cracks, gasps, sobs, sighs, pauses, mood shifts, emphasis, and breath sounds. Anything you would write as a stage direction in a script. The model was trained on film style narration and responds to natural language cues.

**Where to put what.**

- Opening tone or posture goes in the `action_tags` field.
- Mid speech emotional shifts, laughs, and whispers go inline in `[brackets]` inside `speech_text`.

## Preset dropdown

Twelve production tested voices from the [official demos](https://scenema.ai/audio). Auto-fills all fields when selected.

- Old Male Storyteller (fireside, Southern American)
- Young Woman (breathless discovery)
- Terrified Whisper (hiding)
- Irish Woman, Dry Wit
- Rage to Vulnerability (Italian American, arc)
- Eulogy (grief, extremely slow)
- Terror (sobbing, hyperventilating)
- Villain (laughing menace)
- Rain and Thunder (SFX plus urgent shouting)
- Italian Cooking Show (SFX plus enthusiastic)
- Kid Explaining Dinosaurs (8 year old)
- British Woman, East London Rage

Custom lets you write your own from scratch.

## Voice cloning

Two paths for cloning from a real voice.

1. **Load Audio** (built in ComfyUI node) plus **Scenema Audio VAE Encode**. Connect to Generate's `ref_latent` input.
2. **Scenema Audio Load Audio from URL** (paste a link) plus **VAE Encode**. Connect to `ref_latent`.

Reference clip is capped at 20 seconds (longer does not help quality). SeedVC runs automatically after generation to polish voice identity match.

## Languages

Twelve tested. English, Spanish, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Hindi, Arabic, Swahili. Pick from the dropdown, then write `speech_text` in the target language.

## Links

- [Demos and full article](https://scenema.ai/audio)
- [Model weights](https://huggingface.co/ScenemaAI/scenema-audio)
- [Standalone Docker server](https://github.com/ScenemaAI/scenema-audio)

## License

Node code is MIT. See [LICENSE](LICENSE).

Model weights are released under the [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).
