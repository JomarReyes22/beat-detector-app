import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile

st.title("🎵 Beat Detection & Tempo Estimation")

st.write("""
Upload a short music clip and see the estimated tempo (BPM) and detected beats.
""")

uploaded_file = st.file_uploader("Upload an audio file (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Load audio
    y, sr = librosa.load(audio_path)

    # Beat detection
try:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Show tempo
    st.success(f"Estimated Tempo: {tempo:.2f} BPM")

    # Plot waveform with beat markers
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax)
    ax.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
    ax.set(title='Waveform with Detected Beats')
    ax.legend()
    st.pyplot(fig)

    st.caption("Red lines show the estimated beats in the track.")

except Exception as e:
    st.error("⚠️ Beat detection failed. Make sure you're uploading a valid audio file.")
    st.exception(e)
