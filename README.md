# Parakeet TDT Transcription with ONNX Runtime

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Parakeet TDT** is a high-performance implementation of NVIDIA's [Parakeet TDT 1.1B](https://huggingface.co/nvidia/parakeet-tdt-1.1b) model using [ONNX Runtime](https://onnxruntime.ai/), designed for ultra-fast inference on CPU.

This implementation achieves exceptional real-time speeds, outperforming standard [openai/whisper](https://github.com/openai/whisper) and competing directly with GPU-accelerated [faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementations while running entirely on consumer CPUs. The efficiency is achieved through the architectural advantages of the Token-and-Duration Transducer (TDT) model combined with 8-bit quantization.

## Benchmark

### Parakeet TDT vs Faster Whisper

We compare the performance of **Parakeet TDT (CPU)** against **faster-whisper (GPU & CPU)**.

The metric used is **Speedup Factor** (Audio Duration / Processing Time). Higher is better.

| Implementation | Hardware | Model | Precision | Speedup |
| --- | --- | --- | --- | --- |
| **Parakeet TDT** (Ours) | **CPU** (i7-12700KF) | **TDT 1.1B** | **int8** | **~29.7x** |
| **Parakeet TDT** (Ours) | **CPU** (i7-4790) | **TDT 1.1B** | **int8** | **~17.6x** |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | int8 | 13.2x |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | fp16 | 12.4x |
| faster-whisper | CPU (i7-12700K) | Small | int8 | 7.6x |
| faster-whisper | CPU (i7-12700K) | Small | fp32 | 4.9x |

*   **Parakeet TDT**: Benchmarked on Intel Core i7-12700K with ONNX Runtime INT8.
*   **faster-whisper**: Benchmarks from [official faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper).

### Detailed Parakeet Performance

| Metrics | Result |
| --- | --- |
| **Average Speedup** | **29.7x** |
| **Real Time Factor (RTF)** | **0.033** |
| **Max Speedup** | **~30x** |

## Requirements

*   Python 3.10 or greater
*   [FFmpeg](https://ffmpeg.org/) (Required for audio processing)

### CPU Optimization
For hybrid CPUs (like Intel 12th-14th Gen), performance is significantly improved by pinning the process to Performance cores (P-cores).

## Installation

The recommended way to install is via Conda to manage dependencies and Python version cleanly.

```bash
conda create -n parakeet-onnx python=3.10
conda activate parakeet-onnx
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
pip install -r requirements.txt
```

## Usage

### Start the Server

Parakeet TDT provides an OpenAI-compatible API server.

```bash
conda activate parakeet-onnx
python app.py
```
*   **Port**: 5092
*   **Docs**: [http://127.0.0.1:5092/docs](http://127.0.0.1:5092/docs)

### Client Example (Python)

You can use the standard `openai` Python library to interact with the server.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="sk-no-key-required"
)

audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="parakeet-tdt-0.6b-v3",
  file=audio_file,
  response_format="text"
)

print(transcript)
```

### Web Interface

The server includes a built-in web interface for testing and easy drag-and-drop transcription.
Access it at: **[http://127.0.0.1:5092](http://127.0.0.1:5092)**

## Open WebUI Integration

This project is designed to be a drop-in replacement for OpenAI in **Open WebUI**.

1.  Go to **Open WebUI Settings > Audio**.
2.  Set **STT Engine** to `OpenAI`.
3.  Set **OpenAI Base URL** to `http://127.0.0.1:5092/v1`.
4.  Set **OpenAI API Key** to `sk-no-key-required`.
5.  Set **STT Model** to `parakeet-tdt-0.6b-v3`.
6.  Click **Save**.

Now, all voice interactions in Open WebUI will be transcribed locally by Parakeet TDT at lightning speeds.

## Model details

When running the application, the ONNX models are automatically loaded from the `models/` directory. The primary model used is the **Parakeet TDT 1.1B** converted to ONNX with INT8 quantization, providing the optimal balance of speed and accuracy for English speech recognition.
