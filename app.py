import gradio as gr
import torchaudio
import torch
import os
import tempfile
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
import shutil

# مدل Demucs برای جداسازی صدا
model = HDEMUCS_HIGH_MUSDB.get_model()
model.eval()

# تابع جداسازی صدا
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
        output_files = {}  # برای ذخیره فایل‌های خروجی
        for i, target in enumerate(labels):
            if target in selected_sources:
                out_path = os.path.join(tmpdir, f"{target}.wav")
                torchaudio.save(out_path, sources[0, i], 44100)
                output_files[target] = out_path
                result.append((sources[0, i].numpy(), 44100))
            else:
                result.append(None)

        # بازگشت داده‌های صوتی همراه با نمودار طیفی برای پیش‌نمایش
        spec_plots = []
        for i, source in enumerate(result):
            if source is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                librosa.display.waveshow(source[0], sr=44100, ax=ax)
                ax.set_title(f"{labels[i]} - Waveform")
                plt.close(fig)  # جلوگیری از نمایش خودکار
                spec_plots.append(fig)
            else:
                spec_plots.append(None)

        # بازگشت فایل‌های خروجی و نمودارهای صوتی
        return result, spec_plots, output_files

source_options = ["درام", "بیس", "سایر", "وکال"]
source_map = {
    "درام": "drums",
    "بیس": "bass",
    "سایر": "other",
    "وکال": "vocals"
}

# Wrapper برای فراخوانی تابع اصلی
def wrapper(audio, selected_labels):
    selected_sources = [source_map[label] for label in selected_labels]
    result, spec_plots, output_files = separate_audio(audio, selected_sources)
    
    # ایجاد لینک‌های دانلود برای فایل‌های صوتی
    download_links = {key: gr.File(value, label=f"دانلود {key}") for key, value in output_files.items()}
    return result, spec_plots, download_links

# طراحی رابط کاربری جدید با پیش‌نمایش نمودار و دانلود
iface = gr.Interface(
    fn=wrapper,
    inputs=[
        gr.Audio(type="numpy", label="🎵 فایل صوتی خود را آپلود کنید", elem_id="audio-upload"),
        gr.CheckboxGroup(choices=source_options, value=source_options, label="اجزایی که می‌خواهید جدا شوند را انتخاب کنید", elem_id="source-options")
    ],
    outputs=[
        gr.Audio(label="🥁 درام", elem_id="drums-output"),
        gr.Audio(label="🎸 بیس", elem_id="bass-output"),
        gr.Audio(label="🎹 سایر", elem_id="other-output"),
        gr.Audio(label="🎤 وکال", elem_id="vocals-output"),
        gr.Plot(label="📊 نمودار صوتی برای پیش‌نمایش", elem_id="audio-plot"),
        gr.File(label="دانلود فایل‌ها", elem_id="download-links")  # نمایش لینک‌های دانلود
    ],
    title="🎶 جداسازی اجزای آهنگ - نسخه فارسی",
    description="آهنگ خود را آپلود کنید و اجزای مختلف آن را (وکال، درام، بیس، یا سایر سازها) به صورت جداگانه دریافت کنید. این ابزار از مدل پیشرفته Demucs برای جداسازی حرفه‌ای استفاده می‌کند.",
    theme="huggingface",
    allow_flagging="never",
    layout="horizontal",
    live=True
)

# شروع برنامه
iface.launch(server_name="0.0.0.0", server_port=7860)
