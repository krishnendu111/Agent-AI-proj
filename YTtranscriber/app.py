import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

import google.genai as genai
from google.genai import types

# Load environment variables from .env
load_dotenv()

# Configure Gemini API with the new client
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

def generate_gemini_summary(transcript_text: str) -> str:
    """
    Call Gemini to summarize the transcript text using the latest SDK and model.
    """
    prompt = """
You are a YouTube video summarizer.
You will be given the full transcript text of a YouTube video.
Summarize the entire video and provide the important points in bullet form,
within about 200–250 words.

Transcript:
"""
    full_prompt = prompt + transcript_text
    
    # Use the latest model for text generation (gemini-2.5-flash is efficient for summarization)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=types.Part.from_text(text=full_prompt)
    )
    return response.text

def extract_transcript_text(video_url: str) -> str:
    """
    Given a YouTube video URL, fetch the transcript and return it as a single string.
    """
    try:
        # Extract video ID (handles basic patterns like https://www.youtube.com/watch?v=VIDEO_ID or youtu.be/VIDEO_ID)
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format. Could not extract video ID.")
    except IndexError:
        raise ValueError("Invalid YouTube URL format. Could not extract video ID.")

    try:
        # Use the updated API: Instantiate and fetch
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=["en"])

        # Concatenate all text segments into one long paragraph (use .text instead of ["text"])
        full_text = " ".join([entry.text for entry in transcript])
        return full_text.strip()

    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No English transcript found for this video.")
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {e}")

# --------- STREAMLIT UI ---------

st.title("YouTube Video Summarizer with Gemini")

video_url = st.text_input("Enter YouTube URL")

if video_url:
    # Show thumbnail
    try:
        if "v=" in video_url:
            vid_id_for_thumb = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            vid_id_for_thumb = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            vid_id_for_thumb = None
        if vid_id_for_thumb:
            thumb_url = f"https://img.youtube.com/vi/{vid_id_for_thumb}/0.jpg"
            st.image(thumb_url, caption="Video thumbnail")
    except Exception:
        st.warning("Could not display thumbnail. Check if the URL is correct.")

if st.button("Get Detailed Notes"):
    if not video_url:
        st.error("Please enter a YouTube video URL.")
    else:
        with st.spinner("Fetching transcript and generating summary..."):
            try:
                text = extract_transcript_text(video_url)
                st.success("Transcript fetched successfully!")
                st.text_area("Transcript", text, height=300)

                summary = generate_gemini_summary(text)
                st.markdown("### Summary")
                st.markdown(summary)

            except Exception as e:
                st.error(f"Could not process: {e}")