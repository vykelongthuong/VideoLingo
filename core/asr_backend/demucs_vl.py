import os
import torch
import torchaudio as ta
from rich.console import Console
from rich import print as rprint
from demucs.pretrained import get_model
from demucs.audio import AudioFile, convert_audio, save_audio
from demucs.apply import apply_model
import gc
from core.utils.models import *

def demucs_audio():
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return

    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)

    console.print("🤖 Loading <htdemucs> model...")
    model = get_model('htdemucs')
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    console.print("🎵 Separating audio...")
    # Load audio using demucs AudioFile — returns (channels, samples)
    wav = AudioFile(_RAW_AUDIO_FILE).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels
    )
    # Add batch dimension: (channels, samples) → (1, channels, samples)
    wav = wav.unsqueeze(0).to(device)
    # Apply model: expects (batch, channels, samples), returns (batch, sources, channels, samples)
    with torch.no_grad():
        sources = apply_model(model, wav, shifts=1, overlap=0.25, device=device)
    # Remove batch dim: (1, sources, channels, samples) → (sources, channels, samples)
    sources = sources.squeeze(0).to('cpu')
    wav = wav.to('cpu')

    SOURCES = ['bass', 'drums', 'other', 'vocals']
    outputs = {src: sources[idx] for idx, src in enumerate(SOURCES)}

    kwargs = {"samplerate": model.samplerate, "bitrate": 128, "preset": 2,
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}

    console.print("🎤 Saving vocals track...")
    save_audio(outputs['vocals'].cpu(), _VOCAL_AUDIO_FILE, **kwargs)

    console.print("🎹 Saving background music...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), _BACKGROUND_AUDIO_FILE, **kwargs)

    # Clean up memory
    del sources, outputs, background, wav, model
    gc.collect()
    torch.cuda.empty_cache()

    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_audio()
