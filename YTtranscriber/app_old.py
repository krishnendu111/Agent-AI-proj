import os

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled


import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_gemini_summary(transcript_text: str) -> str:
    """
    Call Gemini Pro to summarize the transcript text.
    """
    prompt = """
You are a YouTube video summarizer.
You will be given the full transcript text of a YouTube video.
Summarize the entire video and provide the important points in bullet form,
within about 200–250 words.

Transcript:
"""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text


def extract_transcript_text(video_url: str) -> str:
    """
    Given a YouTube video URL, fetch the transcript and return it as a single string.
    """


    try:
        # Extract video ID (basic pattern: https://www.youtube.com/watch?v=VIDEO_ID)
        video_id = video_url.split("v=")[1]
    except IndexError:
        raise ValueError("Invalid YouTube URL format. Could not extract video ID.")

    """try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript: {e}") 
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en"]
        )

        full_text = " ".join([item["text"] for item in transcript])
        print(full_text)

    except Exception as e:
        print("Error fetching transcript:", e)"""


    # Concatenate all text segments into one long paragraph
    full_text = ""
    for entry in transcript:
        full_text += " " + entry["text"]

    return full_text.strip()


# --------- STREAMLIT UI ---------

st.title("YouTube Video Summarizer with Gemini Pro")

video_url = st.text_input("Enter YouTube URL")

if video_url:
    # Show thumbnail
    try:
        vid_id_for_thumb = video_url.split("v=")[1]
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
                    video_id = video_url.split("v=")[1].split("&")[0]

                    transcript = YouTubeTranscriptApi.get_transcript(
                        video_id,
                        languages=["en"]
                    )

                    text = " ".join([t["text"] for t in transcript])
                    st.success("Transcript fetched successfully!")
                    st.text_area("Transcript", text, height=300)

                    summary = generate_gemini_summary(text)
                    st.markdown("### Summary")
                    st.markdown(summary)


                except Exception as e:
                    st.error(f"Could not fetch transcript: {e}")