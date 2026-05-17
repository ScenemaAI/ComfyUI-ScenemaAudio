# ComfyUI-ScenemaAudio

Native ComfyUI nodes for [Scenema Audio](https://scenema.ai/audio), an expressive text-to-speech model with zero-shot voice cloning built on the LTX 2.3 audio diffusion transformer.

## Nodes

| Node | Description |
|------|-------------|
| **Scenema Audio Prompt Compiler** | Compiles voice description, speech text, and action tags into the prompt format the model expects |
| **Scenema Audio Model Loader** | Loads the 3.3B audio-only transformer (INT8 or bf16) |
| **Scenema Audio VAE Loader** | Loads the audio VAE encoder and decoder |
| **Scenema Audio Text Encode** | Encodes prompts via Gemma 3 12B (NF4 or bf16) |
| **Scenema Audio Sampler** | 8-step distilled diffusion with optional A2V voice reference |
| **Scenema Audio Decode** | Decodes audio latents to waveform |
| **Scenema Audio VAE Encode** | Encodes reference audio for voice cloning |

## Basic Workflow

```
Prompt Compiler → Text Encode → Sampler → Decode → PreviewAudio
                                  ↑
                    Model Loader ──┘
```

Pre-wired workflow JSON files are included in `example_workflows/`.

## VRAM Requirements

ComfyUI automatically offloads models between nodes, so only one model is on GPU at a time.

| VRAM | Transformer | Gemma | Notes |
|------|------------|-------|-------|
| 6 GB | INT8 (4.9 GB) | GGUF Q4 (~3.5 GB) | Minimum viable |
| 12 GB | INT8 (4.9 GB) | NF4 (~8 GB) | Recommended |
| 24 GB | INT8 (4.9 GB) | NF4 (~8 GB) | All models resident |
| 48 GB | bf16 (9.8 GB) | bf16 (24 GB) | Best quality |

## Installation

### ComfyUI Manager

Search for "Scenema Audio" in ComfyUI Manager and click Install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ScenemaAI/ComfyUI-ScenemaAudio.git
pip install -r ComfyUI-ScenemaAudio/requirements.txt
```

Model weights are downloaded automatically from [HuggingFace](https://huggingface.co/ScenemaAI/scenema-audio) on first use.

## Links

- [All demos + article](https://scenema.ai/audio)
- [Model weights](https://huggingface.co/ScenemaAI/scenema-audio)
- [Standalone Docker server](https://github.com/ScenemaAI/scenema-audio)

## License

MIT License. See [LICENSE](LICENSE).

Model weights are released under the [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE).
