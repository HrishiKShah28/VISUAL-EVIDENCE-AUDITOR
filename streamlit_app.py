import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import cv2
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import traceback

# Import your existing modules with error handling
try:
    from main import (
        transcribe_video_enhanced, 
        extract_text_from_video, 
        compare_transcripts_enhanced,
        fetch_reference_video_enhanced,
        DummyVideo,
        EnhancedVideoDB
    )
except ImportError as e:
    st.error(f"Import error from main.py: {e}")
    st.stop()

try:
    from verdict_generator import generate_verdict
except ImportError:
    st.warning("verdict_generator not available - AI analysis will be skipped")
    generate_verdict = None

try:
    from report_generator import generate_html_report
except ImportError:
    st.warning("report_generator not available - HTML reports will be skipped")
    generate_html_report = None

# Define utility functions before main
def get_video_info(video_path):
    """Extract basic video information"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "resolution": f"{width}x{height}",
            "file_size": os.path.getsize(video_path) / (1024*1024)  # MB
        }
    except Exception as e:
        st.error(f"Error reading video info: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'analysis_complete': False,
        'analysis_results': {},
        'temp_files': [],
        'selected_video': None,
        'videodb_video_obj': None,
        'video_list': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def cleanup_temp_files():
    """Clean up temporary files"""
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
    st.session_state.temp_files = []

def save_uploaded_file(uploaded_file, suffix=""):
    """Save uploaded file to temporary location"""
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, f"{uploaded_file.name}{suffix}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.temp_files.append(file_path)
        return file_path
    return None
def get_video_by_id(video_id, api_key):
    """Get video by ID from VideoDB"""
    try:
        from videodb import connect
        conn = connect(api_key=api_key)
        coll = conn.get_collection()
        videos = coll.get_videos()
        
        for video in videos:
            if video.id == video_id:
                enhanced_video = EnhancedVideoDB(video, conn)
                transcript = enhanced_video.get_transcript()
                if transcript and isinstance(transcript, str) and transcript.strip():
                    st.info(f"📄 Video {video_id}: {len(transcript.split())} words in transcript")
                else:
                    st.warning(f"⚠️ Video {video_id}: No transcript available")
                return enhanced_video
        return None
    except Exception as e:
        st.error(f"Error getting video by ID: {e}")
        return None

import streamlit as st
import os
import tempfile
import base64

def get_video_by_title(title, api_key):
    """Search for video by title in VideoDB"""
    try:
        from videodb import connect
        conn = connect(api_key=api_key)
        coll = conn.get_collection()
        videos = coll.get_videos()
        
        title_lower = title.lower()
        for video in videos:
            video_name = getattr(video, 'title', getattr(video, 'name', video.__dict__.get('title', video.__dict__.get('name', '')))).lower()
            if title_lower in video_name:
                enhanced_video = EnhancedVideoDB(video, conn)
                transcript = enhanced_video.get_transcript()
                if transcript and isinstance(transcript, str) and transcript.strip():
                    st.info(f"📄 Video {video.id}: {len(transcript.split())} words in transcript")
                    with st.expander("Preview transcript", expanded=False):
                        st.text(transcript[:500] + "..." if len(transcript) > 500 else transcript)
                else:
                    st.warning(f"⚠️ Video {video.id}: No transcript available")
                return enhanced_video
        return None
    except Exception as e:
        st.error(f"Error searching video by title: {e}")
        return None

def list_all_videos(api_key):
    """Get all videos from VideoDB with detailed transcript feedback"""
    try:
        from videodb import connect
        conn = connect(api_key=api_key)
        coll = conn.get_collection()
        videos = coll.get_videos()
        
        video_list = []
        for video in videos:
            try:
                enhanced_video = EnhancedVideoDB(video, conn)
                transcript = enhanced_video.get_transcript()
                if transcript and isinstance(transcript, str) and transcript.strip():
                    preview = transcript[:100]
                    word_count = len(transcript.split())
                    st.info(f"📄 Video {video.id}: {word_count} words in transcript")
                else:
                    preview = "No transcript available"
                    word_count = 0
                    st.warning(f"⚠️ Video {video.id}: Failed to retrieve transcript")
                
                video_list.append({
                    'id': video.id,
                    'name': getattr(video, 'name', getattr(video, 'title', f'Video {video.id}')),
                    'transcript_preview': preview,
                    'word_count': word_count,
                    'enhanced_obj': enhanced_video
                })
            except Exception as e:
                st.error(f"❌ Error processing video {video.id}: {str(e)}")
                video_list.append({
                    'id': video.id,
                    'name': f'Video {video.id}',
                    'transcript_preview': f'Error: {str(e)}',
                    'word_count': 0,
                    'enhanced_obj': None
                })
        
        return video_list
    except Exception as e:
        st.error(f"Error listing videos: {e}")
        return []
# Define perform_manual_comparison and display_results as in your original code
def perform_manual_comparison(user_video_path, reference_video, progress_bar, status_text):
    """
    COMPLETELY FIXED: Manual implementation that ensures BOTH videos are analyzed
    This bypasses potential issues in the main.py comparison function
    """
    results = {}
    
    try:
        status_text.text("🚀 Initializing manual video comparison...")
        progress_bar.progress(5)
        
        # Step 1: Validate user video
        if not os.path.exists(user_video_path):
            st.error(f"User video not found: {user_video_path}")
            return None
        
        status_text.text("📹 Getting video information...")
        progress_bar.progress(10)
        
        # Get user video info
        user_info = get_video_info(user_video_path)
        results['user_info'] = user_info if user_info else {
            "duration": 0, "fps": 0, "frame_count": 0, "resolution": "Unknown", "file_size": 0
        }
        
        # Step 2: Handle reference video
        ref_video_path = None
        is_videodb_video = False
        
        if isinstance(reference_video, str):
            ref_video_path = reference_video
            st.info(f"📁 Reference video: Local file ({os.path.basename(ref_video_path)})")
        else:
            is_videodb_video = True
            st.info("📥 Reference video: VideoDB video (downloading...)")
            
            status_text.text("📥 Downloading VideoDB reference video...")
            progress_bar.progress(15)
            
            temp_ref_dir = tempfile.mkdtemp()
            temp_ref_filename = f"videodb_ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            temp_ref_path = os.path.join(temp_ref_dir, temp_ref_filename)
            
            try:
                downloaded_path = reference_video.download(temp_ref_path)
                
                if downloaded_path and os.path.exists(downloaded_path):
                    ref_video_path = downloaded_path
                    st.session_state.temp_files.append(downloaded_path)
                    file_size = os.path.getsize(ref_video_path)
                    st.success(f"✅ VideoDB video downloaded successfully ({file_size:,} bytes)")
                else:
                    st.error("❌ Failed to download VideoDB video")
                    return None
            except Exception as e:
                st.error(f"❌ VideoDB download failed: {e}")
                return None
        
        # Get reference video info
        if ref_video_path and os.path.exists(ref_video_path):
            ref_info = get_video_info(ref_video_path)
            results['ref_info'] = ref_info if ref_info else {
                "duration": 0, "fps": 0, "frame_count": 0, "resolution": "Unknown", "file_size": 0
            }
        else:
            st.error("❌ Reference video path is invalid")
            return None
        
        # Step 3: Transcribe BOTH videos
        status_text.text("🎙️ Transcribing user video...")
        progress_bar.progress(25)
        
        st.info(f"🔊 Transcribing user video: {os.path.basename(user_video_path)}")
        user_transcript = transcribe_video_enhanced(user_video_path)
        user_word_count = len(user_transcript.split()) if user_transcript else 0
        
        progress_bar.progress(40)
        
        st.info(f"🔊 Transcribing reference video: {os.path.basename(ref_video_path)}")
        status_text.text("🎙️ Transcribing reference video...")
        
        ref_transcript = transcribe_video_enhanced(ref_video_path)
        ref_word_count = len(ref_transcript.split()) if ref_transcript else 0
        
        results['user_transcript'] = user_transcript
        results['ref_transcript'] = ref_transcript
        results['user_word_count'] = user_word_count
        results['ref_word_count'] = ref_word_count
        
        st.success(f"✅ Transcription complete: {user_word_count} + {ref_word_count} words")
        
        progress_bar.progress(55)
        
        # Step 4: Extract OCR from BOTH videos
        status_text.text("🔍 Extracting text from user video...")
        st.info(f"👁️ OCR processing user video: {os.path.basename(user_video_path)}")
        user_ocr = extract_text_from_video(user_video_path, frame_skip=5, max_frames=50)
        
        progress_bar.progress(70)
        
        st.info(f"👁️ OCR processing reference video: {os.path.basename(ref_video_path)}")
        status_text.text("🔍 Extracting text from reference video...")
        
        ref_ocr = extract_text_from_video(ref_video_path, frame_skip=5, max_frames=50)
        
        results['user_ocr'] = user_ocr
        results['ref_ocr'] = ref_ocr
        
        st.success(f"✅ OCR complete: {len(user_ocr)} + {len(ref_ocr)} text items")
        
        progress_bar.progress(80)
        
        # Step 5: Compare transcripts
        status_text.text("📊 Comparing transcripts...")
        word_score, seq_score, shared_words, total_words = compare_transcripts_enhanced(
            user_transcript, ref_transcript
        )
        
        results['word_score'] = word_score
        results['seq_score'] = seq_score
        results['overlap_score'] = word_score
        results['sequence_score'] = seq_score
        results['shared_words'] = shared_words
        results['total_words'] = total_words
        
        # Step 6: Compare OCR
        user_ocr_set = set(line.strip() for line in user_ocr if line.strip())
        ref_ocr_set = set(line.strip() for line in ref_ocr if line.strip())
        shared_ocr = user_ocr_set.intersection(ref_ocr_set)
        
        if len(user_ocr_set.union(ref_ocr_set)) > 0:
            ocr_similarity = (len(shared_ocr) / len(user_ocr_set.union(ref_ocr_set))) * 100
        else:
            ocr_similarity = 0.0
        
        results['ocr_user_set'] = user_ocr_set
        results['ocr_ref_set'] = ref_ocr_set
        results['shared_ocr'] = shared_ocr
        results['ocr_similarity'] = ocr_similarity
        
        # Step 7: Calculate overall similarity
        similarity_components = []
        if word_score > 0:
            similarity_components.append(word_score)
        if seq_score > 0:
            similarity_components.append(seq_score)
        if ocr_similarity > 0:
            similarity_components.append(ocr_similarity)
        
        if similarity_components:
            overall_similarity = sum(similarity_components) / len(similarity_components)
        else:
            overall_similarity = 0.0
        
        results['overall_similarity'] = overall_similarity
        
        # Step 8: Generate verdict
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
        
        progress_bar.progress(85)
        
        # Step 9: Generate AI verdict if available
        if generate_verdict and os.getenv("OPENAI_API_KEY"):
            try:
                status_text.text("🤖 Generating AI analysis...")
                ai_verdict = generate_verdict(
                    user_transcript=user_transcript,
                    ref_transcript=ref_transcript,
                    user_text_set=user_ocr_set,
                    ref_text_set=ref_ocr_set,
                    duration=results['user_info']['duration'],
                    frames_processed=len(user_ocr),
                    frame_skip=5,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                results['ai_verdict'] = ai_verdict
            except Exception as e:
                results['ai_verdict'] = f"AI verdict generation failed: {str(e)}"
        else:
            results['ai_verdict'] = "AI analysis not available (missing OpenAI API key)"
        
        # Step 10: Generate HTML report if available
        if generate_html_report:
            try:
                status_text.text("📄 Generating HTML report...")
                report_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                generate_html_report(
                    user_transcript=user_transcript,
                    ref_transcript=ref_transcript,
                    user_ocr=user_ocr,
                    ref_ocr=ref_ocr,
                    word_score=word_score,
                    seq_score=seq_score,
                    shared_words=shared_words,
                    verdict_text=verdict,
                    output_path=report_path
                )
                results['report_path'] = report_path
            except Exception as e:
                st.warning(f"Could not generate HTML report: {e}")
                results['report_path'] = None
        else:
            results['report_path'] = None
        
        progress_bar.progress(100)
        status_text.text("✅ Manual analysis complete!")
        
        st.success("🎉 Both videos analyzed successfully!")
        st.info("📊 **Final Results:**")
        st.info(f"   • User video: {user_word_count} words, {len(user_ocr)} OCR items")
        st.info(f"   • Reference video: {ref_word_count} words, {len(ref_ocr)} OCR items")
        st.info(f"   • Audio similarity: {word_score:.1f}% overlap, {seq_score:.1f}% sequence")
        st.info(f"   • OCR similarity: {ocr_similarity:.1f}%")
        st.info(f"   • Overall similarity: {overall_similarity:.1f}%")
        st.info(f"   • Verdict: {verdict}")
        
        return results
    except Exception as e:
        st.error(f"❌ Manual analysis failed: {str(e)}")
        st.error("Full error traceback:")
        st.code(traceback.format_exc())
        return None

def display_results(results):
    """Display analysis results in a structured format"""
    st.markdown("## 📊 Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Transcript Similarity</h3>
            <h2>{results.get('sequence_score', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Word Overlap</h3>
            <h2>{results.get('overlap_score', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>OCR Similarity</h3>
            <h2>{results.get('ocr_similarity', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Score</h3>
            <h2>{results.get('overall_similarity', 0):.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    verdict = results.get('verdict', '')
    if verdict:
        if 'HIGHLY SIMILAR' in verdict.upper():
            status_class = "status-success"
            status_icon = "✅"
        elif 'DIFFERENT' in verdict.upper():
            status_class = "status-error"
            status_icon = "❌"
        else:
            status_class = "status-warning"
            status_icon = "⚠️"
        
        st.markdown(f"""
        <div class="{status_class}">
            <strong>{status_icon} Analysis Verdict:</strong> {verdict}
        </div>
        """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Charts", "🎙️ Transcripts", "🔤 OCR Results", "🧠 AI Analysis", "📄 Reports"])
    
    with tab1:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Similarity Scores', 'Video Properties', 'Content Distribution', 'Word Counts'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(x=['Transcript', 'Word Overlap', 'OCR', 'Overall'], 
                   y=[results.get('sequence_score', 0), results.get('overlap_score', 0), 
                      results.get('ocr_similarity', 0), results.get('overall_similarity', 0)],
                   name='Similarity %', marker_color=['#667eea', '#764ba2', '#f093fb', '#4ecdc4']),
            row=1, col=1
        )
        
        if results.get('user_info') and results.get('ref_info'):
            fig.add_trace(
                go.Bar(x=['Duration (s)', 'File Size (MB)'],
                       y=[results['user_info']['duration'], results['user_info']['file_size']],
                       name='User Video', marker_color='#667eea'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=['Duration (s)', 'File Size (MB)'],
                       y=[results['ref_info']['duration'], results['ref_info']['file_size']],
                       name='Reference Video', marker_color='#764ba2'),
                row=1, col=2
            )
        
        shared_ocr = results.get('shared_ocr', set())
        ocr_user_set = results.get('ocr_user_set', set())
        ocr_ref_set = results.get('ocr_ref_set', set())
        
        content_labels = ['Shared Content', 'User Unique', 'Reference Unique']
        content_values = [
            len(shared_ocr),
            len(ocr_user_set - ocr_ref_set),
            len(ocr_ref_set - ocr_user_set)
        ]
        
        fig.add_trace(
            go.Pie(labels=content_labels, values=content_values, name="Content"),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=['User Video', 'Reference Video'],
                  y=[results.get('user_word_count', 0), results.get('ref_word_count', 0)],
                  name='Word Count', marker_color=['#667eea', '#764ba2']),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎤 User Video Transcript")
            user_transcript = results.get('user_transcript', '')
            if user_transcript:
                st.text_area("", user_transcript, height=300, key="user_transcript_display")
                st.caption(f"Words: {len(user_transcript.split())}")
            else:
                st.info("No speech detected in user video")
        
        with col2:
            st.subheader("🎤 Reference Video Transcript")
            ref_transcript = results.get('ref_transcript', '')
            if ref_transcript:
                st.text_area("", ref_transcript, height=300, key="ref_transcript_display")
                st.caption(f"Words: {len(ref_transcript.split())}")
            else:
                st.info("No speech detected in reference video")
        
        if results.get('shared_words', 0) > 0:
            st.subheader("🤝 Transcript Analysis")
            st.write(f"**Shared words:** {results.get('shared_words', 0)}")
            st.write(f"**Total unique words:** {results.get('total_words', 0)}")
            st.write(f"**Word overlap score:** {results.get('word_score', 0):.1f}%")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        user_ocr = results.get('user_ocr', [])
        ref_ocr = results.get('ref_ocr', [])
        
        with col1:
            st.subheader("🔍 User Video OCR")
            if user_ocr:
                for i, text in enumerate(user_ocr[:20], 1):
                    st.write(f"{i}. {text}")
                if len(user_ocr) > 20:
                    st.caption(f"... and {len(user_ocr) - 20} more items")
            else:
                st.info("No text detected in user video frames")
        
        with col2:
            st.subheader("🔍 Reference Video OCR")
            if ref_ocr:
                for i, text in enumerate(ref_ocr[:20], 1):
                    st.write(f"{i}. {text}")
                if len(ref_ocr) > 20:
                    st.caption(f"... and {len(ref_ocr) - 20} more items")
            else:
                st.info("No text detected in reference video frames")
        
        shared_ocr = results.get('shared_ocr', set())
        if shared_ocr:
            st.subheader("🤝 Shared Text Elements")
            shared_df = pd.DataFrame(list(shared_ocr), columns=['Shared Text'])
            st.dataframe(shared_df, use_container_width=True)
    
    with tab4:
        st.subheader("🧠 AI Analysis")
        verdict = results.get('verdict', '')
        if verdict:
            st.markdown("### 📋 Analysis Verdict")
            st.markdown(verdict)
        
        ai_verdict = results.get('ai_verdict', '')
        if ai_verdict and ai_verdict != "AI analysis not available (missing OpenAI API key)":
            st.markdown("### 🤖 AI-Generated Analysis")
            st.markdown(ai_verdict)
        else:
            st.info("🔑 Enter OpenAI API key in sidebar for AI-powered analysis")
    
    with tab5:
        st.subheader("📄 Detailed Reports")
        summary_data = {
            "Metric": [
                "Overall Similarity", "Transcript Similarity", "Word Overlap", "OCR Similarity",
                "Shared Words", "User Words", "Reference Words", "Shared OCR Elements",
                "User OCR Items", "Reference OCR Items"
            ],
            "Value": [
                f"{results.get('overall_similarity', 0):.2f}%",
                f"{results.get('sequence_score', 0):.2f}%", 
                f"{results.get('overlap_score', 0):.2f}%",
                f"{results.get('ocr_similarity', 0):.2f}%",
                str(results.get('shared_words', 0)),
                str(results.get('user_word_count', 0)),
                str(results.get('ref_word_count', 0)),
                str(len(results.get('shared_ocr', set()))),
                str(len(results.get('user_ocr', []))),
                str(len(results.get('ref_ocr', [])))
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        report_path = results.get('report_path')
        if report_path and os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button(
                    label="📥 Download HTML Report",
                    data=f.read(),
                    file_name=f"video_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
        
        if st.button("📊 Export Results as CSV"):
            csv_data = pd.DataFrame([{
                "Metric": metric,
                "Value": value
            } for metric, value in zip(summary_data["Metric"], summary_data["Value"])])
            csv_string = csv_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_string,
                file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Visual Evidence Auditor</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced AI-powered video authenticity verification and comparison system**")
    
    # Sidebar configuration (unchanged from previous fix)
    with st.sidebar:
        st.header("⚙️ Configuration")
        openai_key = st.text_input("OpenAI API Key", type="password", 
                                  help="Enter your OpenAI API key for AI analysis")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        
        videodb_key = st.text_input("VideoDB API Key", type="password",
                                   help="Enter your VideoDB API key for reference video search")
        if videodb_key:
            os.environ["VIDEODB_API_KEY"] = videodb_key
            try:
                from videodb import connect
                conn = connect(api_key=videodb_key)
                coll = conn.get_collection()
                st.success("✅ VideoDB connection verified")
            except Exception as e:
                st.error(f"❌ Invalid VideoDB API key: {e}")
        
        st.markdown("---")
        st.subheader("🔧 Analysis Options")
        max_frames = st.slider("Max frames for OCR", 10, 100, 50, 
                              help="Maximum number of frames to process for text extraction")
        frame_skip = st.slider("Frame skip interval", 1, 10, 5,
                              help="Process every Nth frame (higher = faster but less thorough)")
        st.markdown("---")
        if st.button("🗑️ Clear Results", key="sidebar_clear"):
            st.session_state.analysis_complete = False
            st.session_state.analysis_results = {}
            st.session_state.selected_video = None
            st.session_state.videodb_video_obj = None
            st.session_state.video_list = None
            cleanup_temp_files()
            st.rerun()
    
    # Main interface
    if not st.session_state.analysis_complete:
        st.header("📁 Upload Videos")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎬 User Video")
            user_video = st.file_uploader(
                "Upload the video you want to analyze",
                type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
                key="user_video"
            )
            if user_video:
                st.video(user_video)
                user_video_path = save_uploaded_file(user_video, "_user")
                if user_video_path:
                    video_info = get_video_info(user_video_path)
                    if video_info:
                        st.caption(f"📊 Duration: {video_info['duration']:.1f}s | "
                                 f"Resolution: {video_info['resolution']} | "
                                 f"Size: {video_info['file_size']:.1f}MB")
        
        with col2:
            st.subheader("📚 Reference Video")
            ref_method = st.radio("Reference Video Source:", 
                                 ["Upload File", "VideoDB - Direct ID", "VideoDB - Title Search", "VideoDB - Browse All"],
                                 key="ref_method")
            
            reference_video_path = None
            reference_video_obj = None
            
            if ref_method == "Upload File":
                ref_video = st.file_uploader(
                    "Upload the reference video for comparison",
                    type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
                    key="ref_video"
                )
                if ref_video:
                    st.video(ref_video)
                    reference_video_path = save_uploaded_file(ref_video, "_ref")
                    if reference_video_path:
                        video_info = get_video_info(reference_video_path)
                        if video_info:
                            st.caption(f"📊 Duration: {video_info['duration']:.1f}s | "
                                      f"Resolution: {video_info['resolution']} | "
                                      f"Size: {video_info['file_size']:.1f}MB")
            
            elif ref_method == "VideoDB - Direct ID":
                st.info("💡 Most reliable method - use the exact video ID from VideoDB")
                video_id = st.text_input("Video ID", 
                                        placeholder="e.g., vid_abc123xyz",
                                        help="Find this in your VideoDB console",
                                        key="video_id")
                if video_id and videodb_key:
                    if st.button("🎯 Get Video by ID", key="get_by_id"):
                        with st.spinner("Getting video by ID..."):
                            reference_video_obj = get_video_by_id(video_id, videodb_key)
                            if reference_video_obj:
                                st.success(f"✅ Found video: {video_id}")
                                st.session_state.videodb_video_obj = reference_video_obj
                                try:
                                    transcript = reference_video_obj.get_transcript()
                                    word_count = len(transcript.split()) if transcript and transcript.strip() else 0
                                    st.info(f"📄 Video has {word_count} words in transcript")
                                    with st.expander("Preview transcript", expanded=False):
                                        st.text(transcript[:500] + "..." if transcript and len(transcript) > 500 else transcript or "No transcript")
                                except Exception as e:
                                    st.warning(f"⚠️ Could not get transcript preview: {str(e)}")
                                # Add Test Transcription button here
                                if st.session_state.videodb_video_obj and st.button("🧪 Test Transcription", key="test_transcription_direct_id"):
                                    with st.spinner("Testing transcription..."):
                                        try:
                                            video_obj = st.session_state.videodb_video_obj
                                            temp_dir = tempfile.mkdtemp()
                                            temp_path = os.path.join(temp_dir, f"test_{video_obj.id}.mp4")
                                            downloaded_path = video_obj.download(temp_path)
                                            if downloaded_path and os.path.exists(downloaded_path):
                                                transcript = transcribe_video_enhanced(downloaded_path)
                                                if transcript and transcript.strip():
                                                    st.success(f"✅ Manual transcription: {len(transcript.split())} words")
                                                    st.text_area("Transcript Preview", transcript[:500], height=200)
                                                else:
                                                    st.error("❌ Manual transcription failed: No text extracted")
                                                # Clean up
                                                if os.path.exists(downloaded_path):
                                                    os.remove(downloaded_path)
                                                os.rmdir(temp_dir)
                                            else:
                                                st.error("❌ Failed to download video for testing")
                                        except Exception as e:
                                            st.error(f"❌ Transcription test failed: {str(e)}")
                                            st.code(traceback.format_exc())
                            else:
                                st.error(f"❌ Video not found with ID: {video_id}")
            
            elif ref_method == "VideoDB - Title Search":
                st.info("💡 Search by video title/name")
                video_title = st.text_input("Video Title", 
                                        placeholder="Enter exact or partial video title",
                                        key="video_title")
                if video_title and videodb_key:
                    if st.button("🔍 Search by Title", key="search_by_title"):
                        with st.spinner("Searching by title..."):
                            reference_video_obj = get_video_by_title(video_title, videodb_key)
                            if reference_video_obj:
                                st.success(f"✅ Found video with title containing: {video_title}")
                                st.session_state.videodb_video_obj = reference_video_obj
                                try:
                                    # Try multiple attributes for video name
                                    video_name = getattr(reference_video_obj.video_object, 'title', 
                                                    getattr(reference_video_obj.video_object, 'name', 
                                                            reference_video_obj.video_object.__dict__.get('title', 
                                                                    reference_video_obj.video_object.__dict__.get('name', 'Unknown'))))
                                    st.info(f"📺 Selected: {video_name}")
                                    transcript = reference_video_obj.get_transcript()
                                    word_count = len(transcript.split()) if transcript and isinstance(transcript, str) and transcript.strip() else 0
                                    if word_count > 0:
                                        st.info(f"📄 Video has {word_count} words in transcript")
                                        with st.expander("Preview transcript", expanded=False):
                                            st.text(transcript[:500] + "..." if len(transcript) > 500 else transcript)
                                    else:
                                        st.warning(f"⚠️ Video has 0 words in transcript")
                                    # Download button
                                    if st.button("📥 Download Video", key="download_video"):
                                        with st.spinner("Downloading video..."):
                                            try:
                                                temp_dir = tempfile.mkdtemp()
                                                temp_path = os.path.join(temp_dir, f"{video_name.replace(' ', '_')}.mp4")
                                                downloaded_path = reference_video_obj.download(temp_path)
                                                if downloaded_path and os.path.exists(downloaded_path):
                                                    with open(downloaded_path, "rb") as file:
                                                        st.download_button(
                                                            label="📥 Click to Download MP4",
                                                            data=file,
                                                            file_name=f"{video_name.replace(' ', '_')}.mp4",
                                                            mime="video/mp4"
                                                        )
                                                    # Clean up
                                                    os.remove(downloaded_path)
                                                    os.rmdir(temp_dir)
                                                else:
                                                    st.error("❌ Failed to download video")
                                            except Exception as e:
                                                st.error(f"❌ Download failed: {str(e)}")
                                                st.code(traceback.format_exc())
                                except Exception as e:
                                    st.warning(f"⚠️ Could not get transcript: {str(e)}")
                                # Test Transcription button
                                if st.session_state.videodb_video_obj and st.button("🧪 Test Transcription", key="test_transcription_title_search"):
                                    with st.spinner("Testing transcription..."):
                                        try:
                                            video_obj = st.session_state.videodb_video_obj
                                            temp_dir = tempfile.mkdtemp()
                                            temp_path = os.path.join(temp_dir, f"test_{video_obj.id}.mp4")
                                            downloaded_path = video_obj.download(temp_path)
                                            if downloaded_path and os.path.exists(downloaded_path):
                                                transcript = transcribe_video_enhanced(downloaded_path)
                                                if transcript and isinstance(transcript, str) and transcript.strip():
                                                    st.success(f"✅ Manual transcription: {len(transcript.split())} words")
                                                    st.text_area("Transcript Preview", transcript[:500], height=200)
                                                else:
                                                    st.error("❌ Manual transcription failed: No text extracted")
                                                # Clean up
                                                if os.path.exists(downloaded_path):
                                                    os.remove(downloaded_path)
                                                os.rmdir(temp_dir)
                                            else:
                                                st.error("❌ Failed to download video for testing")
                                        except Exception as e:
                                            st.error(f"❌ Transcription test failed: {str(e)}")
                                            st.code(traceback.format_exc())
                            else:
                                st.error(f"❌ No video found with title: {video_title}")
                        
            elif ref_method == "VideoDB - Browse All":
                st.info("💡 Browse and select from all your videos")
                if videodb_key:
                    if st.button("📋 Load All Videos", key="load_all"):
                        with st.spinner("Loading your videos..."):
                            video_list = list_all_videos(videodb_key)
                            if video_list:
                                st.session_state.video_list = video_list
                                st.success(f"✅ Loaded {len(video_list)} videos")
                            else:
                                st.error("❌ No videos found in your collection")
                    
                    if st.session_state.video_list:
                        st.subheader("Select Video:")
                        video_options = []
                        for video in st.session_state.video_list:
                            preview = video['transcript_preview'][:50] + "..." if len(video['transcript_preview']) > 50 else video['transcript_preview']
                            option_text = f"{video['name']} ({video['word_count']} words) - {preview}"
                            video_options.append(option_text)
                        
                        selected_index = st.selectbox(
                            "Choose a video:",
                            range(len(video_options)),
                            format_func=lambda x: f"{x+1}. {video_options[x]}",
                            key="video_selection"
                        )
                        
                        if st.button("✅ Select This Video", key="select_video"):
                            selected_video_info = st.session_state.video_list[selected_index]
                            if selected_video_info.get('enhanced_obj'):
                                reference_video_obj = selected_video_info['enhanced_obj']
                                st.success(f"✅ Selected: {selected_video_info['name']}")
                                st.session_state.videodb_video_obj = reference_video_obj
                            else:
                                reference_video_obj = get_video_by_id(selected_video_info['id'], videodb_key)
                                if reference_video_obj:
                                    st.success(f"✅ Selected: {selected_video_info['name']}")
                                    st.session_state.videodb_video_obj = reference_video_obj
                                else:
                                    st.error("❌ Could not load selected video")
                            # Add Test Transcription button here
                            if st.session_state.videodb_video_obj and st.button("🧪 Test Transcription", key="test_transcription_browse_all"):
                                with st.spinner("Testing transcription..."):
                                    try:
                                        video_obj = st.session_state.videodb_video_obj
                                        temp_dir = tempfile.mkdtemp()
                                        temp_path = os.path.join(temp_dir, f"test_{video_obj.id}.mp4")
                                        downloaded_path = video_obj.download(temp_path)
                                        if downloaded_path and os.path.exists(downloaded_path):
                                            transcript = transcribe_video_enhanced(downloaded_path)
                                            if transcript and transcript.strip():
                                                st.success(f"✅ Manual transcription: {len(transcript.split())} words")
                                                st.text_area("Transcript Preview", transcript[:500], height=200)
                                            else:
                                                st.error("❌ Manual transcription failed: No text extracted")
                                            # Clean up
                                            if os.path.exists(downloaded_path):
                                                os.remove(downloaded_path)
                                            os.rmdir(temp_dir)
                                        else:
                                            st.error("❌ Failed to download video for testing")
                                    except Exception as e:
                                        st.error(f"❌ Transcription test failed: {str(e)}")
                                        st.code(traceback.format_exc())
        
        # Rest of the main function (Analysis button, results display, etc.) remains unchanged
        # ...
        
        # Analysis button
        st.markdown("---")
        user_video_ready = user_video is not None
        ref_video_ready = (reference_video_path is not None) or (st.session_state.videodb_video_obj is not None)
        
        if user_video_ready and ref_video_ready:
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True, key="start_analysis"):
                if ref_method == "Upload File":
                    final_ref = reference_video_path
                else:
                    final_ref = st.session_state.videodb_video_obj
                
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                results = perform_manual_comparison(
                    user_video_path, final_ref, 
                    progress_bar, status_text
                )
                
                if results:
                    st.session_state.analysis_results = results
                    st.session_state.analysis_complete = True
                    st.rerun()
        else:
            if not user_video_ready:
                st.info("👆 Please upload a user video to start")
            elif not ref_video_ready:
                st.info("👆 Please select a reference video to start the analysis")
    
    else:
        display_results(st.session_state.analysis_results)
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Start New Analysis", key="new_analysis"):
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = {}
                st.session_state.selected_video = None
                st.session_state.videodb_video_obj = None
                st.session_state.video_list = None
                cleanup_temp_files()
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All Data", key="clear_all"):
                cleanup_temp_files()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# Page configuration (move to top if not already)
st.set_page_config(
    page_title="Visual Evidence Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()