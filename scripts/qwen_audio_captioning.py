"""
Qwen2-Audio-7B-Instruct ESC-50 Captioning Script
Minimal example for public release (NeurIPS paper code)
"""
import torch
import soundfile as sf
import librosa
import tempfile
import os
import io
import argparse

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from huggingface_hub import login

def main():
    parser = argparse.ArgumentParser(description="Audio captioning with Qwen2-Audio-7B-Instruct on ESC-50.")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help='Model name')
    parser.add_argument('--prompt', type=str, default="Describe the sounds in this audio clip.", help='Prompt for captioning')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token (or set HF_TOKEN env var)')
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ds = load_dataset("ashraq/esc50", split="train", streaming=True)
    example = next(iter(ds))
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with io.BytesIO(example["audio"]) as audio_buffer:
                array, sr = sf.read(audio_buffer)
                if sr != 16000:
                    array = librosa.resample(array, orig_sr=sr, target_sr=16000)
                    sr = 16000
                sf.write(tmp.name, array, samplerate=sr)
                audio_path = tmp.name

        inputs = processor(
            audio=audio_path,
            text=args.prompt,
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Filename: {example['filename']}")
        print(f"Category: {example['category']}")
        print(f"Caption: {caption}")
        os.remove(audio_path)
    except Exception as e:
        print(f"Error processing example: {e}")

if __name__ == "__main__":
    main()
