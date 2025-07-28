# Visual Evidence Auditor

An AI powered Streamlit web application for analyzing and comparing videos using VideoDB for video management, faster-whisper for audio transcription, and OCR for text extraction from video frames.

## Features
- Search and download videos from VideoDB by title or semantic search.
- Transcribe video audio using faster-whisper with ffmpeg for audio extraction.
- Compare videos based on audio transcripts and on-screen text via OCR.
- Generate verdicts on video similarity (e.g., "HIGHLY SIMILAR").
- User-friendly Streamlit interface for video selection and result visualization.

## Installation

## Dependencies
See `requirements.txt` for Python packages and `packages.txt` for system dependencies:
- Python: streamlit==1.39.0, videodb, faster-whisper, ffmpeg-python==0.2.0, python-dotenv==1.0.1, requests==2.32.3, pytesseract==0.3.13, imutils==0.5.4, tqdm==4.66.5, numpy==1.26.4, opencv-python==4.8.1.78, pandas==2.2.3, plotly==5.24.1
- System: ffmpeg, tesseract-ocr, libgl1, libglib2.0-0, libatlas-base-dev, libopenblas-dev
### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/visual-evidence-auditor.git
   cd visual-evidence-auditor
