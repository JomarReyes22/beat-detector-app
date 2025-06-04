import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import io

# --- Page Configuration ---
st.set_page_config(page_title="🎵 Beat Detection App", layout="centered")

# --- Custom Title Section ---
st.markdown("<h1 style='text-align: center;'>🎧 Beat Detection & Tempo Estimation</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center;">
Upload a short audio file to:
<ul style="text-align: left; display: inline-block;">
    <li>📊 Estimate the tempo (BPM)</li>
    <li>📍 See beat positions on the waveform</li>
    <li>🌈 View a frequency spectrogram</li>
    <li>▶️ Listen to your uploaded audio</li>
    <li>📤 Download beat timestamps</li>
</ul>
</div>
""", unsafe_allow_html=True)

# --- File Upload ---
st.markdown("### 🎵 Upload Your Audio File")
uploaded_file = st.file_uploader("", type=["wav", "mp3"])

if uploaded_file:
    st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

    # Display audio player
    st.markdown("### ▶️ Audio Preview")
    st.audio(uploaded_file, format="audio/wav")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Load audio
    y, sr = librosa.load(audio_path)

    try:
        # --- Beat Detection ---
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        st.markdown(f"<h3 style='color: green;'>✅ Estimated Tempo: {float(tempo):.2f} BPM</h3>", unsafe_allow_html=True)

        # --- Waveform with Beats ---
        st.markdown("### 📈 Waveform with Beat Markers")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax1)
        ax1.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
        ax1.set(title="Waveform")
        ax1.legend()
        st.pyplot(fig1)

        # --- Spectrogram ---
        st.markdown("### 🌈 Log-frequency Spectrogram")
        stft = np.abs(librosa.stft(y))
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                       sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set(title="Spectrogram")
        st.pyplot(fig2)

        # --- Download Button ---
        st.markdown("### 📤 Export Beat Times")
        beat_output = "\n".join([f"{t:.3f} sec" for t in beat_times])
        st.download_button(
            label="📥 Download beat timestamps as .txt",
            data=beat_output,
            file_name="beat_timestamps.txt",
            mime="text/plain"
        )

        st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Beat detection failed. Please try another file.")
        st.exception(e)
else:
    st.markdown("<p style='text-align:center; color: gray;'>No file uploaded yet.</p>", unsafe_allow_html=True)
