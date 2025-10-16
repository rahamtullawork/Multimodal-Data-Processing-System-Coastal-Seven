# av_transcriber.py
import os
import tempfile
import subprocess
import whisper
import yt_dlp


def transcribe_media(file_path_or_url):
    """
    Transcribes audio/video files or YouTube links into text.
    Supported: .mp3, .mp4, YouTube URLs
    Returns recognized text as string.
    """
    temp_dir = tempfile.gettempdir()
    audio_path = None
    text_output = ""

    try:
        # 1️⃣ Handle YouTube URLs
        if str(file_path_or_url).startswith("http"):
            stt_model = _load_whisper_model()
            audio_path = download_youtube_audio(file_path_or_url, temp_dir)
            text_output = transcribe_audio(audio_path, stt_model)
            return text_output

        # 2️⃣ Handle local MP3 / MP4
        ext = os.path.splitext(str(file_path_or_url))[1].lower()
        if ext in [".mp3", ".mp4"]:
            stt_model = _load_whisper_model()
            # Convert to wav if needed
            audio_path = convert_to_wav(file_path_or_url, temp_dir)
            text_output = transcribe_audio(audio_path, stt_model)
            return text_output

        else:
            print(f"[WARN] Unsupported file type: {file_path_or_url}")
            return ""

    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return ""
    finally:
        # Clean up temporary audio file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


# -------------------------------
# Helper Functions
# -------------------------------

def _load_whisper_model():
    """Load base Whisper model (fast and light)."""
    return whisper.load_model("base")  # or "base" or "tiny" for even lighter model


def convert_to_wav(input_path, output_dir):
    """Convert MP3/MP4 to WAV using ffmpeg."""
    output_path = os.path.join(output_dir, "audio_temp.wav")
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", str(input_path),
        "-ar", "16000",  # sample rate
        "-ac", "1",  # mono
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def transcribe_audio(audio_path, model):
    """Run Whisper model to transcribe audio."""
    result = model.transcribe(audio_path)
    return result["text"].strip()


def download_youtube_audio(url, output_dir):
    """Download audio from YouTube video."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "yt_audio.%(ext)s"),
        "quiet": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    mp3_path = os.path.join(output_dir, "yt_audio.mp3")
    return mp3_path
