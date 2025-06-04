import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile

# --- Color and layout setup ---
dark_bg = "#0e1117"
light_gray = "#d0d0d0"
accent_color = "#00c6a9"

st.set_page_config(page_title="🎧 Beat Detection App", layout="centered")

# --- Title Section ---
st.markdown(
    f"""
    <div style='background-color: {dark_bg}; padding: 30px; border-radius: 10px;'>
        <h1 style='text-align: center; color: {accent_color}; font-size: 40px;'>🎶 Beat Detection & Tempo Estimation</h1>
        <p style='text-align: center; color: {light_gray}; font-size: 16px;'>
            Upload an audio file to analyze its tempo, see waveform beats, and visualize sound with a spectrogram.
        </p>
        <p style='text-align: center; font-size: 24px;'>🔊🎧🎵</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Upload Section ---
st.markdown("### 🔊 Upload Your Audio File")
uploaded_file = st.file_uploader("Drop your .wav or .mp3 file here", type=["wav", "mp3"])

if uploaded_file:
    st.markdown("---")
    
    # --- Audio Player ---
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

        # Display Tempo
        st.markdown(
            f"<h3 style='color:{accent_color};'>✅ Estimated Tempo: {float(tempo):.2f} BPM</h3>",
            unsafe_allow_html=True
        )

        # --- Waveform Plot ---
        st.markdown("### 📈 Waveform with Beat Markers")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax1)
        ax1.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
        ax1.set(title="Waveform")
        ax1.legend()
        st.pyplot(fig1)

        # --- Spectrogram Plot ---
        st.markdown("### 🌈 Spectrogram View")
        stft = np.abs(librosa.stft(y))
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                       sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set(title="Log-frequency Spectrogram")
        st.pyplot(fig2)

        # --- Download Beat Times ---
        st.markdown("### 📤 Download Beat Timestamps")
        beat_output = "\n".join([f"{t:.3f} sec" for t in beat_times])
        st.download_button(
            label="📥 Download beat timestamps as .txt",
            data=beat_output,
            file_name="beat_timestamps.txt",
            mime="text/plain"
        )

        st.markdown("---")

    except Exception as e:
        st.error("❌ Beat detection failed. Please try another audio file.")
        st.exception(e)

else:
    st.markdown("<p style='text-align:center; color: gray;'>🔇 No audio file uploaded yet.</p>", unsafe_allow_html=True)
