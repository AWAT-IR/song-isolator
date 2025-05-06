import gradio as gr
import torchaudio
import torch
import os
import tempfile
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

model = HDEMUCS_HIGH_MUSDB.get_model()
model.eval()

def separate_audio(audio, selected_sources):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "input.wav")
        torchaudio.save(audio_path, torch.tensor(audio[0]), sample_rate=audio[1])
        waveform, sr = torchaudio.load(audio_path)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        with torch.inference_mode():
            sources = model(waveform.unsqueeze(0))
        result = []
        labels = ["drums", "bass", "other", "vocals"]
        for i, target in enumerate(labels):
            if target in selected_sources:
                out_path = os.path.join(tmpdir, f"{target}.wav")
                torchaudio.save(out_path, sources[0, i], 44100)
                result.append((sources[0, i].numpy(), 44100))
            else:
                result.append(None)
        return result

source_options = ["درام", "بیس", "سایر", "وکال"]
source_map = {
    "درام": "drums",
    "بیس": "bass",
    "سایر": "other",
    "وکال": "vocals"
}

def wrapper(audio, selected_labels):
    selected_sources = [source_map[label] for label in selected_labels]
    return separate_audio(audio, selected_sources)

iface = gr.Interface(
    fn=wrapper,
    inputs=[
        gr.Audio(type="numpy", label="🎵 فایل صوتی خود را آپلود کنید"),
        gr.CheckboxGroup(choices=source_options, value=source_options, label="اجزایی که می‌خواهید جدا شوند را انتخاب کنید")
    ],
    outputs=[
        gr.Audio(label="🥁 درام"),
        gr.Audio(label="🎸 بیس"),
        gr.Audio(label="🎹 سایر"),
        gr.Audio(label="🎤 وکال")
    ],
    title="🎶 جداسازی اجزای آهنگ - نسخه فارسی",
    description="آهنگ خود را آپلود کنید و اجزای مختلف آن را (وکال، درام، بیس، یا سایر سازها) به صورت جداگانه دریافت کنید. این ابزار از مدل پیشرفته Demucs برای جداسازی حرفه‌ای استفاده می‌کند.",
    theme="huggingface",
    allow_flagging="never"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
