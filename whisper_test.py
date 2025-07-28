from faster_whisper import WhisperModel

# Load model (uses float32 if GPU float16 unsupported)
model = WhisperModel("Systran/faster-whisper-small", compute_type="int8", device="cpu")


segments, info = model.transcribe("sample.mp4", beam_size=5)

print(f"Detected language: {info.language}")
print("--- Transcript ---")

# Save transcript to file
with open("audio_text.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        line = f"[{segment.start:.2f} - {segment.end:.2f}]  {segment.text.strip()}"
        print(line)
        f.write(segment.text.strip() + "\n")
