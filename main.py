import os
import sys
import difflib
import subprocess
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import time
import pytesseract
import cv2
import numpy as np
from imutils.video import FileVideoStream
from tqdm import tqdm
from difflib import SequenceMatcher
import hashlib
import json
import tempfile
import shutil
import requests
from urllib.parse import urlparse
import logging
from videodb import connect
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract path (adjust as needed for your system)
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except:
    # For systems where Tesseract is in PATH
    pass

# Load environment variables
load_dotenv()

def check_ffmpeg():
    """Check if FFmpeg is available and working"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
        else:
            print("❌ FFmpeg check failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg not found in PATH")
        return False

def extract_audio_with_ffmpeg(video_path, output_audio_path):
    """Extract audio from video using FFmpeg - MORE RELIABLE"""
    try:
        print(f"🎵 Extracting audio from {os.path.basename(video_path)}...")
        
        # Use FFmpeg to extract audio with better compatibility
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', video_path,
            '-vn',  # no video
            '-acodec', 'pcm_s16le',  # uncompressed audio
            '-ar', '16000',  # 16kHz sample rate (Whisper's preferred)
            '-ac', '1',  # mono
            '-f', 'wav',  # WAV format
            output_audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_audio_path):
            file_size = os.path.getsize(output_audio_path)
            print(f"✅ Audio extracted successfully ({file_size:,} bytes)")
            return True
        else:
            print(f"❌ FFmpeg failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Audio extraction timed out")
        return False
    except Exception as e:
        print(f"❌ Audio extraction error: {e}")
        return False

def transcribe_video_enhanced(video_path):
    """Enhanced video transcription with multiple fallback methods"""
    if not whisper_model:
        print("❌ Whisper model not available")
        return ""
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return ""
    
    # Validate file size first
    file_size = os.path.getsize(video_path)
    if file_size < 10000:  # Less than 10KB is likely not a valid video
        print(f"❌ File too small ({file_size} bytes) - likely not a valid video")
        return ""
    
    print(f"🎙️ Enhanced transcription for: {os.path.basename(video_path)} ({file_size:,} bytes)")
    
    # Method 1: Try with FFmpeg audio extraction first
    try:
        print("🔄 Method 1: FFmpeg + Whisper...")
        
        # Create temporary audio file
        temp_dir = tempfile.mkdtemp()
        temp_audio = os.path.join(temp_dir, "temp_audio.wav")
        
        if extract_audio_with_ffmpeg(video_path, temp_audio):
            # Transcribe the extracted audio
            segments, info = whisper_model.transcribe(
                temp_audio,
                beam_size=1,
                word_timestamps=False,
                vad_filter=False,
                language=None
            )
            
            transcript_parts = []
            for segment in segments:
                text = segment.text.strip() if hasattr(segment, 'text') else str(segment).strip()
                if text and len(text) > 1:
                    transcript_parts.append(text)
            
            # Cleanup
            try:
                os.remove(temp_audio)
                os.rmdir(temp_dir)
            except:
                pass
            
            if transcript_parts:
                transcript = " ".join(transcript_parts)
                word_count = len(transcript.split())
                print(f"✅ Method 1 success: {word_count} words")
                return transcript
                
        # Cleanup on failure
        try:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            os.rmdir(temp_dir)
        except:
            pass
            
    except Exception as e:
        print(f"⚠️ Method 1 failed: {e}")
    
    # Method 2: Direct transcription with basic settings
    try:
        print("🔄 Method 2: Direct Whisper transcription...")
        segments, info = whisper_model.transcribe(
            video_path, 
            beam_size=1,
            word_timestamps=False,
            vad_filter=False,
            language=None
        )
        
        transcript_parts = []
        for segment in segments:
            text = segment.text.strip() if hasattr(segment, 'text') else str(segment).strip()
            if text and len(text) > 1:
                transcript_parts.append(text)
        
        if transcript_parts:
            transcript = " ".join(transcript_parts)
            word_count = len(transcript.split())
            print(f"✅ Method 2 success: {word_count} words")
            return transcript
            
    except Exception as e:
        print(f"⚠️ Method 2 failed: {e}")
    
    # Method 3: With relaxed settings (fixed parameters)
    try:
        print("🔄 Method 3: Relaxed settings...")
        segments, info = whisper_model.transcribe(
            video_path,
            beam_size=1,
            word_timestamps=False,
            vad_filter=False,
            language=None,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6
        )
        
        transcript_parts = []
        for segment in segments:
            text = segment.text.strip() if hasattr(segment, 'text') else str(segment).strip()
            if text:
                transcript_parts.append(text)
        
        if transcript_parts:
            transcript = " ".join(transcript_parts)
            word_count = len(transcript.split())
            print(f"✅ Method 3 success: {word_count} words")
            return transcript
            
    except Exception as e:
        print(f"⚠️ Method 3 failed: {e}")
    
    print("❌ All transcription methods failed")
    return ""

def try_multiple_transcript_methods(video_object):
    """Try different ways to get transcript from VideoDB"""
    
    methods = [
        ("get_transcript()", lambda v: v.get_transcript()),
        ("transcript attribute", lambda v: getattr(v, 'transcript', '')),
        ("get_spoken_words()", lambda v: getattr(v, 'get_spoken_words', lambda: '')()),
        ("segments text", lambda v: extract_from_segments(v)),
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"   Trying {method_name}...")
            result = method_func(video_object)
            
            if result and isinstance(result, str) and len(result.strip()) > 10:
                print(f"   ✅ {method_name} succeeded!")
                return result.strip()
            elif result:
                print(f"   ⚠️ {method_name} returned short result: '{str(result)[:50]}'")
            else:
                print(f"   ❌ {method_name} returned empty")
                
        except Exception as e:
            print(f"   ❌ {method_name} failed: {e}")
    
    return ""

def extract_from_segments(video_object):
    """Try to extract text from video segments or timeline"""
    try:
        # Try different segment methods
        if hasattr(video_object, 'get_segments'):
            segments = video_object.get_segments()
            texts = []
            for segment in segments:
                if hasattr(segment, 'text'):
                    texts.append(segment.text)
                elif hasattr(segment, 'transcript'):
                    texts.append(segment.transcript)
            return " ".join(texts)
            
        if hasattr(video_object, 'timeline'):
            timeline = video_object.timeline
            if hasattr(timeline, 'get_text'):
                return timeline.get_text()
                
    except Exception as e:
        print(f"   Segment extraction error: {e}")
    
    return ""

def download_videodb_video_enhanced(video_object, output_dir="reference_videos"):
    """Enhanced VideoDB video download with better error handling"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Generate unique filename
        timestamp = int(time.time())
        output_filename = f"videodb_ref_{timestamp}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"📥 Downloading VideoDB video to: {output_path}")
        
        # Method 1: Try get_stream_url or similar methods
        stream_url = None
        url_methods = ['get_stream_url', 'stream_url', 'url', 'download_url', 'video_url', 'file_url']
        
        for method_name in url_methods:
            try:
                if hasattr(video_object, method_name):
                    attr = getattr(video_object, method_name)
                    if callable(attr):
                        stream_url = attr()
                    else:
                        stream_url = attr
                    
                    if stream_url and isinstance(stream_url, str) and stream_url.startswith('http'):
                        print(f"✅ Found stream URL via {method_name}: {stream_url[:50]}...")
                        break
                    else:
                        stream_url = None
            except Exception as e:
                print(f"   ⚠️ {method_name} failed: {e}")
                continue
        
        if stream_url:
            try:
                print("🔄 Downloading from stream URL...")
                
                # Add headers for better compatibility
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(stream_url, stream=True, timeout=60, headers=headers)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    
                    # Validate minimum file size
                    if file_size > 10000:  # At least 10KB
                        print(f"✅ Stream download successful ({file_size:,} bytes)")
                        
                        # Verify with OpenCV
                        cap = cv2.VideoCapture(output_path)
                        if cap.isOpened():
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            
                            if frame_count > 0:
                                print(f"✅ Video verification passed ({frame_count} frames, {fps:.1f} FPS)")
                                return output_path
                            else:
                                print("⚠️ Video has no frames")
                        else:
                            print("⚠️ Downloaded file cannot be opened by OpenCV")
                            cap.release()
                    else:
                        print(f"⚠️ Downloaded file too small ({file_size} bytes)")
                        
            except Exception as e:
                print(f"⚠️ Stream download failed: {e}")
        
        # Method 2: Try standard download method if exists
        try:
            if hasattr(video_object, 'download'):
                print("🔄 Trying standard download method...")
                downloaded_path = video_object.download(output_path)
                
                if downloaded_path and os.path.exists(downloaded_path):
                    file_size = os.path.getsize(downloaded_path)
                    
                    if file_size > 10000:
                        print(f"✅ Standard download successful ({file_size:,} bytes)")
                        return downloaded_path
                    else:
                        print(f"⚠️ Standard download file too small ({file_size} bytes)")
                        
        except Exception as e:
            print(f"⚠️ Standard download failed: {e}")
        
        # Method 3: Check for direct file methods
        file_methods = ['get_file', 'download_file', 'export', 'save']
        for method_name in file_methods:
            try:
                if hasattr(video_object, method_name):
                    print(f"🔄 Trying {method_name}...")
                    method = getattr(video_object, method_name)
                    if callable(method):
                        result = method(output_path)
                        if result and os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            if file_size > 10000:
                                print(f"✅ {method_name} successful ({file_size:,} bytes)")
                                return output_path
                            
            except Exception as e:
                print(f"⚠️ {method_name} failed: {e}")
                continue
        
        print("❌ All download methods failed")
        return None
        
    except Exception as e:
        print(f"❌ Enhanced download error: {e}")
        return None
def fetch_reference_video_enhanced(search_term: str, conn, output_dir="reference_videos"):
    """FIXED: Enhanced VideoDB integration with proper connection handling"""
    if not conn:
        print("❌ Error: VideoDB connection (conn) is None")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from videodb import SearchType, IndexType
        
        print(f"🔐 FIXED: Connecting to VideoDB...")
        print("✅ Connected successfully")
        
        # Get collection
        try:
            coll = conn.get_collection()
            print(f"📂 Retrieved collection")
        except Exception as e:
            print(f"❌ Collection error: {e}")
            return None
        
        # Get videos
        videos = coll.get_videos()
        if not videos:
            print("❌ No videos found")
            return None
        
        print(f"📹 Found {len(videos)} videos")
        
        # First try: Search by title (exact match)
        print(f"🔍 Searching for video with title: '{search_term}'...")
        for video in videos:
            try:
                video_title = getattr(video, 'title', '') or getattr(video, 'name', '') or ''
                if video_title and video_title.lower().strip() == search_term.lower().strip():
                    print(f"✅ Found exact title match: '{video_title}'")
                    return EnhancedVideoDB(video, conn)  # Ensure conn is passed
            except Exception as e:
                print(f"   ⚠️ Error checking video title: {e}")
                continue
        
        # Second try: Semantic search with proper error handling
        print(f"🔍 Performing semantic search for: '{search_term}'...")
        
        shots = []
        try:
            results = coll.search(
                query=search_term,
                search_type=SearchType.semantic,
                index_type=IndexType.spoken_word,
                limit=3
            )
            shots = results.get_shots()
            print(f"✅ Search found {len(shots)} results")
        except Exception as e:
            print(f"⚠️ Semantic search failed: {e}")
        
        # Return best result with conn parameter
        if shots:
            video = coll.get_video(shots[0].video_id)
            video_title = getattr(video, 'title', 'Untitled')
            print(f"✅ Using search result: '{video_title}'")
            return EnhancedVideoDB(video, conn)  # Ensure conn is passed
        elif videos:
            print("⚠️ Using first available video as fallback")
            fallback_title = getattr(videos[0], 'title', 'Untitled')
            print(f"   Fallback video: '{fallback_title}'")
            return EnhancedVideoDB(videos[0], conn)  # Ensure conn is passed
        
        return None
        
    except Exception as e:
        print(f"❌ VideoDB error: {e}")
        import traceback
        traceback.print_exc()
        return None
def search_video_by_title(search_term: str, conn):
    """Enhanced title-based video search with fuzzy matching - FIXED VERSION"""
    if not conn:
        print("❌ Error: VideoDB connection (conn) is None")
        return None

    try:
        from difflib import SequenceMatcher
        
        print(f"🔍 Enhanced title search for: '{search_term}'")
        
        coll = conn.get_collection()
        videos = coll.get_videos()
        
        if not videos:
            print("❌ No videos in collection")
            return None
        
        print(f"📊 Checking {len(videos)} videos for title matches...")
        
        # Try exact matches first
        for video in videos:
            try:
                title = getattr(video, 'title', '') or getattr(video, 'name', '') or ''
                if title and title.lower().strip() == search_term.lower().strip():
                    print(f"✅ Exact match found: '{title}'")
                    return EnhancedVideoDB(video, conn)  # Ensure conn is passed
            except Exception as e:
                print(f"   Error checking exact match: {e}")
                continue
        
        # Try fuzzy matching with similarity threshold
        print("🔄 No exact matches, trying fuzzy matching...")
        best_match = None
        best_similarity = 0.6  # Minimum 60% similarity
        
        for video in videos:
            try:
                title = getattr(video, 'title', '') or getattr(video, 'name', '') or ''
                if title:
                    similarity = SequenceMatcher(None, 
                                               search_term.lower().strip(), 
                                               title.lower().strip()).ratio()
                    
                    print(f"   '{title}' -> {similarity:.2f} similarity")
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = video
                        print(f"   ✅ New best match: '{title}' ({similarity:.2f})")
                        
            except Exception as e:
                print(f"   Error checking fuzzy match: {e}")
                continue
        
        if best_match:
            title = getattr(best_match, 'title', 'Untitled')
            print(f"✅ Best fuzzy match selected: '{title}' ({best_similarity:.2f} similarity)")
            return EnhancedVideoDB(best_match, conn)  # Ensure conn is passed
        else:
            print(f"❌ No good matches found for '{search_term}' (minimum similarity: 0.6)")
            return None
            
    except Exception as e:
        print(f"❌ Title search error: {e}")
        import traceback
        traceback.print_exc()
        return None
import os
import requests
import traceback

import os
import requests
import tempfile
import traceback

import os
import subprocess
import tempfile
import traceback
import requests

import os
import tempfile
import traceback

import os
import tempfile
import traceback
import subprocess

class EnhancedVideoDB:
    def __init__(self, video_object, conn):
        self.video_object = video_object
        self.id = getattr(video_object, 'id', 'unknown')
        self.conn = conn
        self.downloaded_path = None
        self._cached_transcript = None

    def download(self, path=None):
        """Download the video from VideoDB using conn.download or ffmpeg fallback"""
        if self.downloaded_path and os.path.exists(self.downloaded_path):
            print(f"✅ Using cached download: {self.downloaded_path}")
            return self.downloaded_path

        try:
            # Create a temporary path if none provided
            if path is None:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, f"videodb_{self.id}.mp4")
            
            print(f"📥 Downloading video {self.id} to: {path}")
            # Try conn.download with M3U8 stream URL
            stream_url = f"https://stream.videodb.io/v3/published/manifests/{self.id}.m3u8"
            print(f"🔗 Stream URL: {stream_url}")
            try:
                response = self.conn.download(stream_link=stream_url, name=path)
                print(f"📊 Download response: {response}")
            except Exception as e:
                print(f"❌ conn.download failed: {str(e)}")
                # Fallback: Check video object for stream URL
                video_attrs = self.video_object.__dict__
                print(f"🔍 Video object attributes: {video_attrs}")
                stream_url = video_attrs.get('stream_url', video_attrs.get('url', stream_url))
                print(f"🔗 Fallback stream URL: {stream_url}")
                response = self.conn.download(stream_link=stream_url, name=path)
                print(f"📊 Fallback download response: {response}")

            # Validate downloaded file
            if os.path.exists(path) and os.path.getsize(path) > 1024 * 1024:  # At least 1MB
                self.downloaded_path = path
                print(f"✅ Download successful: {self.downloaded_path} ({os.path.getsize(path)} bytes)")
                return self.downloaded_path
            else:
                print(f"⚠️ Downloaded file invalid or too small: {path}")
                os.remove(path) if os.path.exists(path) else None
                self.downloaded_path = None
                # Try ffmpeg fallback
                print("🔄 Falling back to ffmpeg download...")
                cmd = [
                    "ffmpeg", "-headers", f"Authorization: Bearer {self.conn.api_key}",
                    "-i", stream_url, "-c", "copy", "-bsf:a", "aac_adtstoasc", path
                ]
                print(f"📥 Running ffmpeg command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"📊 ffmpeg stdout: {result.stdout}")
                print(f"📊 ffmpeg stderr: {result.stderr}")
                if os.path.exists(path) and os.path.getsize(path) > 1024 * 1024:
                    self.downloaded_path = path
                    print(f"✅ ffmpeg download successful: {self.downloaded_path} ({os.path.getsize(path)} bytes)")
                    return self.downloaded_path
                else:
                    print(f"⚠️ ffmpeg downloaded file invalid: {path}")
                    os.remove(path) if os.path.exists(path) else None
                    return None
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            traceback.print_exc()
            self.downloaded_path = None
            return None

    def get_transcript(self):
        """Get transcript from VideoDB using get_transcript_text or local transcription"""
        if self._cached_transcript is not None:
            print(f"✅ Returning cached transcript: {len(self._cached_transcript.split())} words")
            return self._cached_transcript
        
        try:
            print(f"🔍 Attempting to fetch transcript for video ID: {self.id}")
            # Try get_transcript_text() first
            transcript = self.video_object.get_transcript_text()
            if transcript and isinstance(transcript, str) and transcript.strip():
                print(f"✅ VideoDB transcript retrieved: {len(transcript.split())} words")
                self._cached_transcript = transcript
                return transcript
            else:
                print(f"⚠️ VideoDB get_transcript_text empty or invalid: {transcript}")
        except Exception as e:
            print(f"❌ VideoDB get_transcript_text error: {str(e)}")
            traceback.print_exc()
        
        # Try get_transcript() as fallback
        try:
            transcript_json = self.video_object.get_transcript()
            if transcript_json and isinstance(transcript_json, list) and transcript_json:
                transcript = " ".join(segment.get('text', '') for segment in transcript_json if segment.get('text'))
                if transcript.strip():
                    print(f"✅ VideoDB get_transcript retrieved: {len(transcript.split())} words")
                    self._cached_transcript = transcript
                    return transcript
                else:
                    print(f"⚠️ VideoDB get_transcript empty: {transcript_json}")
            else:
                print(f"⚠️ VideoDB get_transcript empty or invalid: {transcript_json}")
        except Exception as e:
            print(f"❌ VideoDB get_transcript error: {str(e)}")
            traceback.print_exc()

        # Fallback to local transcription
        try:
            print("🔄 Falling back to local transcription...")
            video_path = self.download()
            if video_path and os.path.exists(video_path):
                print(f"📥 Video downloaded to: {video_path}")
                transcript = transcribe_video_enhanced(video_path)
                if transcript and isinstance(transcript, str) and transcript.strip():
                    print(f"✅ Local transcript retrieved: {len(transcript.split())} words")
                    self._cached_transcript = transcript
                    return transcript
                else:
                    print(f"⚠️ Local transcript empty: {transcript}")
                    self._cached_transcript = ""
                    return ""
            else:
                print("❌ Failed to download video for transcription")
                self._cached_transcript = ""
                return ""
        except Exception as e:
            print(f"❌ Local transcription error: {str(e)}")
            traceback.print_exc()
            self._cached_transcript = ""
            return ""
def compare_transcripts_enhanced(t1, t2):
    """Enhanced transcript comparison with better debugging"""
    print(f"\n🔍 ENHANCED TRANSCRIPT COMPARISON:")
    print(f"   Input 1 length: {len(str(t1)) if t1 else 0} chars")
    print(f"   Input 2 length: {len(str(t2)) if t2 else 0} chars")
    
    # Handle None, empty, or invalid inputs
    if not t1 or not t2:
        print("   ❌ One or both transcripts are empty/None")
        return 0.0, 0.0, 0, 0
    
    try:
        transcript1 = str(t1).strip().lower()
        transcript2 = str(t2).strip().lower()
    except:
        print("   ❌ Could not convert inputs to strings")
        return 0.0, 0.0, 0, 0
    
    if not transcript1 or not transcript2:
        print("   ❌ One or both transcripts are empty after conversion")
        return 0.0, 0.0, 0, 0
    
    try:
        # Normalize and split into words
        words1 = transcript1.split()
        words2 = transcript2.split()
        
        print(f"   📊 Word counts: {len(words1)} vs {len(words2)}")
        
        if len(words1) == 0 or len(words2) == 0:
            print("   ❌ One transcript has no words after splitting")
            return 0.0, 0.0, 0, 0
        
        # Word overlap calculation (improved)
        word_set1 = set(words1)
        word_set2 = set(words2)
        
        # Count actual word overlaps
        overlap_count = 0
        for word in word_set1:
            if word in word_set2:
                count1 = words1.count(word)
                count2 = words2.count(word)
                overlap_count += min(count1, count2)
        
        # Calculate overlap percentage based on average length
        avg_length = (len(words1) + len(words2)) / 2
        overlap_score = (overlap_count / avg_length) * 100 if avg_length > 0 else 0
        
        # Sequence similarity using difflib
        sequence_similarity = difflib.SequenceMatcher(None, transcript1, transcript2).ratio() * 100
        
        # Total unique words
        total_unique_words = len(word_set1.union(word_set2))
        
        print(f"   ✅ Enhanced results: {overlap_score:.2f}% overlap, {sequence_similarity:.2f}% sequence")
        print(f"   📊 {overlap_count} shared words, {total_unique_words} total unique")
        
        return overlap_score, sequence_similarity, overlap_count, total_unique_words
        
    except Exception as e:
        print(f"   ❌ Enhanced comparison error: {e}")
        return 0.0, 0.0, 0, 0

def preprocess_frame(frame):
    """Preprocess frame for better OCR results"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return gray
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return frame

def is_similar(a, b, threshold=0.85):
    """Check if two strings are similar"""
    return SequenceMatcher(None, a, b).ratio() > threshold

def extract_text_from_video(video_path, frame_skip=5, max_frames=50, save_path=None):
    """Extract text from video frames using OCR with enhanced error handling"""
    print(f"\n🎥 Processing video for OCR: {os.path.basename(video_path)}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Failed to open video file for OCR")
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("❌ Failed to read FPS from video")
            cap.release()
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"📊 Video info: {duration:.2f}s at {fps:.1f} FPS ({total_frames} total frames)")

        print(f"🔍 Processing every {frame_skip} frames (max {max_frames})")
        sampled_frames = []

        for i in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            sampled_frames.append(frame)
            if len(sampled_frames) >= max_frames:
                break

        cap.release()

        print(f"🧠 Processing {len(sampled_frames)} frames for text...")
        ocr_results = []

        for frame in tqdm(sampled_frames, desc="📸 OCR Processing"):
            try:
                preprocessed = preprocess_frame(frame)
                raw_text = pytesseract.image_to_string(preprocessed, config="--psm 6")
                cleaned = raw_text.strip()
                if len(cleaned) > 3 and any(c.isalnum() for c in cleaned):
                    ocr_results.append(cleaned)
            except Exception as e:
                continue

        # Deduplicate similar entries
        final_results = []
        for txt in ocr_results:
            if all(not is_similar(txt, existing) for existing in final_results):
                final_results.append(txt)

        print(f"✅ Unique OCR Results: {len(final_results)} items")
        
        # Save to file if requested
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    for line in final_results:
                        f.write(line + "\n")
                print(f"📁 OCR results saved to: {save_path}")
            except Exception as e:
                print(f"⚠️ Could not save OCR results: {e}")

        return final_results
        
    except Exception as e:
        print(f"❌ Error extracting text from video: {e}")
        return []

def perform_enhanced_comparison(user_video_path, reference_video):
    """FIXED: Enhanced comparison with proper analysis of BOTH videos"""
    print("\n" + "="*60)
    print("🚀 STARTING ENHANCED VIDEO COMPARISON")
    print("="*60)
    
    # Validate user video
    if not os.path.exists(user_video_path):
        print(f"❌ User video not found: {user_video_path}")
        return create_error_result("User video file not found")
    
    print(f"📁 User video: {os.path.basename(user_video_path)}")
    
    # FIXED: Completely revamped reference video handling
    ref_video_path = None
    is_videodb = False
    
    if isinstance(reference_video, str):
        # It's a local file path
        ref_video_path = reference_video  
        print(f"📁 Reference video (local): {os.path.basename(ref_video_path)}")
        if not os.path.exists(ref_video_path):
            print(f"❌ Reference video not found: {ref_video_path}")
            return create_error_result("Reference video file not found")
    else:
        # It's a VideoDB object - FORCE complete download process
        print("📥 Processing VideoDB reference video...")
        is_videodb = True
        
        # Create output directory
        output_dir = "reference_videos"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        temp_filename = f"videodb_force_{timestamp}.mp4"
        temp_path = os.path.join(output_dir, temp_filename)
        
        print(f"🔄 Forcing fresh download to: {temp_path}")
        
        # Try multiple download approaches
        download_success = False
        
        # Method 1: Use enhanced download method directly
        try:
            print("   Trying enhanced download method...")
            ref_video_path = download_videodb_video_enhanced(reference_video.video_object, output_dir)
            
            if ref_video_path and os.path.exists(ref_video_path):
                file_size = os.path.getsize(ref_video_path)
                if file_size > 10000:  # At least 10KB
                    print(f"   ✅ Method 1 success: {file_size:,} bytes")
                    download_success = True
                else:
                    print(f"   ❌ Method 1 failed - file too small ({file_size} bytes)")
            else:
                print(f"   ❌ Method 1 failed - no file created")
        except Exception as e:
            print(f"   ❌ Method 1 failed: {e}")
        
        # Method 2: Try direct stream download if available
        if not download_success:
            try:
                print("   Trying direct stream download...")
                
                # Check if video object has stream URL
                stream_url = None
                url_attrs = ['get_stream_url', 'stream_url', 'url', 'download_url', 'video_url']
                
                for attr_name in url_attrs:
                    try:
                        if hasattr(reference_video.video_object, attr_name):
                            attr = getattr(reference_video.video_object, attr_name)
                            if callable(attr):
                                stream_url = attr()
                            else:
                                stream_url = attr
                            
                            if stream_url and isinstance(stream_url, str) and stream_url.startswith('http'):
                                break
                        stream_url = None
                    except:
                        continue
                
                if stream_url:
                    print(f"   Found stream URL: {stream_url[:50]}...")
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    response = requests.get(stream_url, stream=True, timeout=60, headers=headers)
                    response.raise_for_status()
                    
                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    if os.path.exists(temp_path):
                        file_size = os.path.getsize(temp_path)
                        if file_size > 10000:
                            print(f"   ✅ Method 2 success: {file_size:,} bytes")
                            ref_video_path = temp_path
                            download_success = True
                        else:
                            print(f"   ❌ Method 2 failed - file too small ({file_size} bytes)")
                    else:
                        print(f"   ❌ Method 2 failed - no file created")
                else:
                    print("   ❌ Method 2 failed - no stream URL found")
                    
            except Exception as e:
                print(f"   ❌ Method 2 failed: {e}")
        
        # Final check
        if not download_success or not ref_video_path or not os.path.exists(ref_video_path):
            print("❌ ALL DOWNLOAD METHODS FAILED")
            print(f"   Attempted path: {temp_path}")
            print(f"   File exists: {os.path.exists(temp_path) if temp_path else False}")
            
            # Debug: List VideoDB object attributes
            print("🔍 DEBUG: VideoDB object attributes:")
            for attr in dir(reference_video.video_object):
                if not attr.startswith('_'):
                    try:
                        value = getattr(reference_video.video_object, attr)
                        if not callable(value):
                            print(f"   {attr}: {str(value)[:100]}")
                    except:
                        print(f"   {attr}: <unable to read>")
            
            return create_error_result("Reference video download completely failed")
        
        print(f"📁 Reference video (downloaded): {os.path.basename(ref_video_path)}")
    
    # Video file validation
    print("\n🔍 VIDEO FILE VALIDATION:")
    print("-" * 40)
    
    # Check file sizes
    try:
        user_size = os.path.getsize(user_video_path)
        ref_size = os.path.getsize(ref_video_path)
        
        print(f"User video: {user_size:,} bytes ({user_size/1024/1024:.2f} MB)")
        print(f"Reference video: {ref_size:,} bytes ({ref_size/1024/1024:.2f} MB)")
        
        if user_size == 0:
            return create_error_result("User video file is empty")
        if ref_size == 0 or ref_size < 10000:
            return create_error_result(f"Reference video file is too small ({ref_size} bytes) - likely corrupted")
            
    except Exception as e:
        print(f"❌ File size check failed: {e}")
        return create_error_result(f"File access error: {str(e)}")
    
    # OpenCV compatibility check
    print("\n🎥 VIDEO COMPATIBILITY CHECK:")
    print("-" * 40)
    
    user_video_ok = ref_video_ok = False
    user_duration = ref_duration = 0
    
    try:
        cap1 = cv2.VideoCapture(user_video_path)
        user_video_ok = cap1.isOpened()
        if user_video_ok:
            user_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            user_fps = cap1.get(cv2.CAP_PROP_FPS)
            user_duration = user_frames / user_fps if user_fps > 0 else 0
            print(f"✅ User video: {user_frames} frames, {user_fps:.1f} FPS, {user_duration:.1f}s")
        else:
            print("❌ User video: Cannot be opened by OpenCV")
        cap1.release()
        
        cap2 = cv2.VideoCapture(ref_video_path)
        ref_video_ok = cap2.isOpened()
        if ref_video_ok:
            ref_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            ref_fps = cap2.get(cv2.CAP_PROP_FPS)
            ref_duration = ref_frames / ref_fps if ref_fps > 0 else 0
            print(f"✅ Reference video: {ref_frames} frames, {ref_fps:.1f} FPS, {ref_duration:.1f}s")
        else:
            print("❌ Reference video: Cannot be opened by OpenCV")
        cap2.release()
        
    except Exception as e:
        print(f"❌ Video compatibility check failed: {e}")
        return create_error_result(f"Video format error: {str(e)}")
    
    if not user_video_ok:
        return create_error_result("User video format not supported")
    if not ref_video_ok:
        return create_error_result("Reference video format not supported or corrupted")
    
    # FIXED: Force transcription of BOTH videos using file paths only
    print("\n🎙️ FORCED AUDIO TRANSCRIPTION:")
    print("-" * 40)
    
    print("🔄 Transcribing user video (Whisper)...")
    user_transcript = transcribe_video_enhanced(user_video_path)
    user_word_count = len(user_transcript.split()) if user_transcript else 0
    
    print(f"📊 User transcript: {user_word_count} words")
    if user_word_count > 0:
        preview = user_transcript[:100] + "..." if len(user_transcript) > 100 else user_transcript
        print(f"   Preview: '{preview}'")
    else:
        print("   ⚠️ No audio content detected in user video")
    
    print(f"\n🔄 Transcribing reference video (Whisper): {os.path.basename(ref_video_path)}")
    
    # CRITICAL FIX: ALWAYS use Whisper on the downloaded file path
    # Completely bypass VideoDB transcript methods to ensure consistency
    ref_transcript = transcribe_video_enhanced(ref_video_path)
    ref_word_count = len(ref_transcript.split()) if ref_transcript else 0
    
    print(f"📊 Reference transcript: {ref_word_count} words")
    if ref_word_count > 0:
        preview = ref_transcript[:100] + "..." if len(ref_transcript) > 100 else ref_transcript
        print(f"   Preview: '{preview}'")
    else:
        print("   ⚠️ No audio content detected in reference video")
    
    # FIXED: Enhanced transcript comparison with proper error handling
    print("\n📊 ENHANCED TRANSCRIPT COMPARISON:")
    print("-" * 40)
    
    if user_word_count == 0 and ref_word_count == 0:
        print("❌ Both transcripts are empty - no audio content detected")
        word_score = seq_score = shared_words = total_words = 0
    elif user_word_count == 0:
        print("❌ User video transcript is empty - cannot compare")
        word_score = seq_score = shared_words = total_words = 0
    elif ref_word_count == 0:
        print("❌ Reference video transcript is empty - cannot compare")
        word_score = seq_score = shared_words = total_words = 0
    else:
        print("✅ Both transcripts available - performing comparison...")
        word_score, seq_score, shared_words, total_words = compare_transcripts_enhanced(
            user_transcript, ref_transcript
        )
    
    # FIXED: Force OCR processing for BOTH videos using exact file paths
    print("\n👁️ FORCED OCR TEXT EXTRACTION:")
    print("-" * 40)
    
    # Process user video OCR
    print(f"🔄 Extracting text from user video: {os.path.basename(user_video_path)}")
    user_ocr = []
    try:
        user_ocr = extract_text_from_video(user_video_path, frame_skip=15, max_frames=10)
        print(f"📊 User OCR: {len(user_ocr)} text items extracted")
        if user_ocr:
            print(f"   Sample: {user_ocr[0][:50]}..." if len(user_ocr[0]) > 50 else f"   Sample: {user_ocr[0]}")
    except Exception as e:
        print(f"⚠️ User OCR failed: {e}")
    
    # CRITICAL FIX: Force OCR on reference video using exact file path
    print(f"\n🔄 Extracting text from reference video: {os.path.basename(ref_video_path)}")
    ref_ocr = []
    try:
        ref_ocr = extract_text_from_video(ref_video_path, frame_skip=15, max_frames=10)
        print(f"📊 Reference OCR: {len(ref_ocr)} text items extracted")
        if ref_ocr:
            print(f"   Sample: {ref_ocr[0][:50]}..." if len(ref_ocr[0]) > 50 else f"   Sample: {ref_ocr[0]}")
        else:
            print("   ⚠️ No OCR text extracted from reference video")
    except Exception as e:
        print(f"⚠️ Reference OCR failed: {e}")
    
    # OCR comparison (basic)
    ocr_similarity = 0.0
    if user_ocr and ref_ocr:
        user_ocr_text = " ".join(user_ocr).lower()
        ref_ocr_text = " ".join(ref_ocr).lower()
        ocr_similarity = SequenceMatcher(None, user_ocr_text, ref_ocr_text).ratio() * 100
        print(f"📊 OCR Text Similarity: {ocr_similarity:.1f}%")
    elif not user_ocr and not ref_ocr:
        print("📊 No OCR text found in either video")
    elif not user_ocr:
        print("📊 No OCR text found in user video")
    elif not ref_ocr:
        print("📊 No OCR text found in reference video")
    
    # Final results
    print("\n✅ ENHANCED COMPARISON COMPLETE:")
    print("-" * 40)
    print(f"📊 Audio Similarity: {word_score:.1f}% word overlap, {seq_score:.1f}% sequence match")
    print(f"📊 Shared words: {shared_words}, Total unique: {total_words}")
    print(f"📊 OCR Similarity: {ocr_similarity:.1f}%")
    print(f"📊 OCR items: {len(user_ocr)} user, {len(ref_ocr)} reference")
    
    # Create comprehensive results
    results = {
        'word_score': float(word_score),
        'seq_score': float(seq_score),
        'shared_words': int(shared_words),
        'total_words': int(total_words),
        'ocr_similarity': float(ocr_similarity),
        'user_transcript': str(user_transcript) if user_transcript else "",
        'ref_transcript': str(ref_transcript) if ref_transcript else "",
        'user_ocr': user_ocr if user_ocr else [],
        'ref_ocr': ref_ocr if ref_ocr else [],
        'user_word_count': user_word_count,
        'ref_word_count': ref_word_count,
        'user_exists': True,
        'ref_exists': True,
        'user_size': user_size,
        'ref_size': ref_size,
        'processing_success': True,
        'user_duration': user_duration,
        'ref_duration': ref_duration
    }
    
    # Calculate overall similarity score
    overall_similarity = 0.0
    similarity_components = []
    
    if word_score > 0:
        similarity_components.append(word_score)
    if seq_score > 0:
        similarity_components.append(seq_score)
    if ocr_similarity > 0:
        similarity_components.append(ocr_similarity)
    
    if similarity_components:
        overall_similarity = sum(similarity_components) / len(similarity_components)
        results['overall_similarity'] = overall_similarity
        print(f"📊 Overall Similarity: {overall_similarity:.1f}%")
    else:
        results['overall_similarity'] = 0.0
        print("📊 Overall Similarity: 0.0% (no comparable content found)")
    
    # Generate verdict
    verdict_parts = []
    
    if word_score > 70:
        verdict_parts.append("HIGH audio similarity")
    elif word_score > 40:
        verdict_parts.append("MODERATE audio similarity")
    elif word_score > 10:
        verdict_parts.append("LOW audio similarity")
    else:
        verdict_parts.append("NO significant audio similarity")
    
    if ocr_similarity > 70:
        verdict_parts.append("HIGH visual text similarity")
    elif ocr_similarity > 40:
        verdict_parts.append("MODERATE visual text similarity")
    elif ocr_similarity > 10:
        verdict_parts.append("LOW visual text similarity")
    else:
        verdict_parts.append("NO significant visual text similarity")
    
    if overall_similarity > 70:
        verdict = f"HIGHLY SIMILAR videos: {', '.join(verdict_parts)}"
    elif overall_similarity > 40:
        verdict = f"MODERATELY SIMILAR videos: {', '.join(verdict_parts)}"
    elif overall_similarity > 10:
        verdict = f"SLIGHTLY SIMILAR videos: {', '.join(verdict_parts)}"
    else:
        verdict = f"DIFFERENT videos: {', '.join(verdict_parts)}"
    
    results['verdict'] = verdict
    print(f"\n🧾 Final Verdict: {verdict}")
    
    return results

def create_error_result(error_message):
    """Create a standardized error result dictionary"""
    return {
        'word_score': 0.0,
        'seq_score': 0.0,
        'shared_words': 0,
        'total_words': 0,
        'ocr_similarity': 0.0,
        'overall_similarity': 0.0,
        'user_transcript': "",
        'ref_transcript': "",
        'user_ocr': [],
        'ref_ocr': [],
        'user_word_count': 0,
        'ref_word_count': 0,
        'user_exists': False,
        'ref_exists': False,
        'user_size': 0,
        'ref_size': 0,
        'processing_success': False,
        'error': error_message,
        'verdict': f"Error: {error_message}",
        'user_duration': 0,
        'ref_duration': 0
    }

def get_video_title(video_path):
    """Extract title from video file path"""
    return os.path.splitext(os.path.basename(video_path))[0]

class DummyVideo:
    """Fallback class for manual video paths"""
    def __init__(self, path):
        self.path = path
        self.id = os.path.basename(path)
    
    def download(self, output_path=None):
        """Copy video to output path or return original path"""
        if output_path:
            try:
                shutil.copy(self.path, output_path)
                return output_path
            except Exception as e:
                print(f"Error copying video: {e}")
        return self.path
    
    def get_transcript(self):
        """Get transcript using enhanced Whisper"""
        return transcribe_video_enhanced(self.path)

def get_user_and_reference_videos_enhanced():
    """Enhanced function to get user video and reference video - FIXED VERSION"""
    print("🚀 Enhanced Video Comparison System Initialization...")

    # Step 1: Get user video
    user_video = input("📁 Enter path to your video file: ").strip()
    if not os.path.exists(user_video):
        print("❌ User video file not found.")
        return None, None

    print(f"📱 User video: {os.path.basename(user_video)}")
    
    # Step 2: Get search term
    search_title = get_video_title(user_video)
    print(f"🔍 Default search term: '{search_title}'")
    custom_term = input(f"🎯 Press Enter to use '{search_title}' or type a different search term: ").strip()
    search_term = custom_term if custom_term else search_title
    
    # Step 3: Try VideoDB with improved search
    print("\n" + "="*50)
    print("🌐 Attempting VideoDB reference video fetch...")

    api_key = os.getenv("VIDEODB_API_KEY")
    reference_video = None

    if api_key:
        try:
            from videodb import connect
            conn = connect(api_key=api_key)
            print("✅ VideoDB connection established")
            
            # Debug: Verify connection
            coll = conn.get_collection()
            videos = coll.get_videos()
            print(f"📊 Available videos: {len(videos)}")
            
            # Try title-based search first  
            print(f"🔄 Trying title-based search for: '{search_term}'...")
            reference_video = search_video_by_title(search_term, conn)
            
            # Fallback to semantic search
            if reference_video is None:
                print("🔄 Title search failed, trying semantic search...")
                reference_video = fetch_reference_video_enhanced(search_term, conn)
            
            if reference_video is not None:
                print("✅ VideoDB search successful!")
            else:
                print("❌ All VideoDB search methods failed")
                
        except Exception as e:
            print(f"❌ Could not connect to VideoDB: {e}")
            import traceback
            traceback.print_exc()
            reference_video = None
    else:
        print("⚠️ VIDEODB_API_KEY not found in environment")

    # Step 4: Manual fallback
    if reference_video is None:
        print("\n📁 VideoDB fetch failed or unavailable")
        print("Available options:")
        print("1. Enter path to a local reference video file")
        print("2. Skip and continue without reference video")
        
        choice = input("📁 Enter path to reference video manually (or press Enter to skip): ").strip()
        
        if choice and os.path.exists(choice):
            reference_video = DummyVideo(choice)
            print(f"✅ Using manual reference: {os.path.basename(choice)}")
        else:
            print("❌ No valid reference video provided.")
            return None, None

    print(f"✅ Reference video ready for comparison")
    return user_video, reference_video

def debug_videodb_connection(api_key):
    """Debug function to test VideoDB connection and list all videos"""
    try:
        print("🔍 DEBUG: Testing VideoDB connection...")
        
        from videodb import connect
        conn = connect(api_key=api_key)
        print("✅ Connection successful")
        
        coll = conn.get_collection()
        print("✅ Collection retrieved")
        
        videos = coll.get_videos()
        print(f"✅ Found {len(videos)} videos")
        
        print("\n📊 All available videos:")
        print("-" * 60)
        
        for i, video in enumerate(videos, 1):
            try:
                video_id = getattr(video, 'id', 'unknown')
                title = getattr(video, 'title', '') or getattr(video, 'name', '') or 'Untitled'
                duration = getattr(video, 'duration', 'unknown')
                
                print(f"{i:2d}. ID: {video_id}")
                print(f"    Title: '{title}'")
                print(f"    Duration: {duration}")
                
                # Show all attributes for debugging
                print("    Attributes:", end=" ")
                attrs = []
                for attr in dir(video):
                    if not attr.startswith('_') and not callable(getattr(video, attr)):
                        try:
                            value = getattr(video, attr)
                            if value and str(value) not in ['unknown', '', 'None']:
                                attrs.append(f"{attr}={str(value)[:30]}")
                        except:
                            pass
                print(", ".join(attrs[:3]))  # Show first 3 non-empty attributes
                print()
                
            except Exception as e:
                print(f"{i:2d}. Error reading video info: {e}")
                print()
        
        return conn, videos
        
    except Exception as e:
        print(f"❌ Debug connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []
def test_search_fix():
    """Test function to verify the search fix works"""
    print("🧪 Testing search fix...")
    
    api_key = os.getenv("VIDEODB_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return False
    
    try:
        conn, videos = debug_videodb_connection(api_key)
        if not conn or not videos:
            print("❌ Connection test failed")
            return False
        
        # Test with a real video title
        if videos:
            test_title = getattr(videos[0], 'title', '') or 'test'
            print(f"🧪 Testing search with title: '{test_title}'")
            
            result = search_video_by_title(test_title, conn)
            if result:
                print("✅ Search fix verified - function works correctly!")
                return True
            else:
                print("❌ Search still failing")
                return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return False
def main_enhanced():
    """Enhanced main function with comprehensive error handling"""
    print("🚀 FIXED VISUAL EVIDENCE AUDITOR")
    print("="*60)
    
    try:
        # Get videos
        user_video_path, reference_video = get_user_and_reference_videos_enhanced()
        
        if user_video_path is None or reference_video is None:
            print("❌ Could not obtain required video files.")
            return create_error_result("Required video files not available")

        # Perform enhanced comparison
        results = perform_enhanced_comparison(user_video_path, reference_video)
        
        # Display summary
        print("\n" + "="*60)
        print("📊 FINAL COMPARISON SUMMARY")
        print("="*60)
        
        if results['processing_success']:
            print(f"✅ Processing completed successfully")
            print(f"📊 Audio Word Overlap: {results['word_score']:.1f}%")
            print(f"📊 Audio Sequence Match: {results['seq_score']:.1f}%")
            print(f"📊 OCR Text Similarity: {results['ocr_similarity']:.1f}%")
            print(f"📊 Overall Similarity: {results['overall_similarity']:.1f}%")
            print(f"📊 Shared Words: {results['shared_words']}")
            print(f"📊 User Transcript: {results['user_word_count']} words")
            print(f"📊 Reference Transcript: {results['ref_word_count']} words")
            print(f"📊 OCR Extractions: {len(results['user_ocr'])} + {len(results['ref_ocr'])}")
            print(f"📊 Video Durations: {results['user_duration']:.1f}s + {results['ref_duration']:.1f}s")
            print(f"\n🎯 VERDICT: {results['verdict']}")
        else:
            print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        return create_error_result("Process interrupted")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return create_error_result(f"Unexpected error: {str(e)}")

# Initialize Whisper model with enhanced error handling
try:
    print("🔄 Loading Whisper model...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading Whisper model: {e}")
    print("⚠️ Trying alternative Whisper model...")
    try:
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Alternative Whisper model loaded")
    except Exception as e2:
        print(f"❌ All Whisper models failed: {e2}")
        whisper_model = None

# Setup directories
UPLOAD_DIR = "uploads"
REFERENCE_DIR = "reference_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)

# Export functions for Streamlit import
__all__ = [
    'transcribe_video_enhanced',
    'extract_text_from_video', 
    'compare_transcripts_enhanced',
    'fetch_reference_video_enhanced',
    'EnhancedVideoDB',
    'DummyVideo',
    'get_video_title',
    'preprocess_frame',
    'is_similar',
    'perform_enhanced_comparison',
    'create_error_result',
    'get_user_and_reference_videos_enhanced',
    'check_ffmpeg',
    'extract_audio_with_ffmpeg',
    'download_videodb_video_enhanced',
    'try_multiple_transcript_methods',
    'main_enhanced'
]

# Backward compatibility aliases
transcribe_video = transcribe_video_enhanced
compare_transcripts = compare_transcripts_enhanced
fetch_reference_video = fetch_reference_video_enhanced
perform_comparison = perform_enhanced_comparison
get_user_and_reference_videos = get_user_and_reference_videos_enhanced
main = main_enhanced

if __name__ == "__main__":
    main_enhanced()