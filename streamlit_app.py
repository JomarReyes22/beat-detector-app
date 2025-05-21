import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import io

st.set_page_config(page_title="🎵 Beat Detection Prototype", layout="centered")

st.title("🎵 Beat Detection & Tempo Estimation with Audio & Spectrogram")
st.markdown("""
Upload a short audio file (.wav or .mp3) to:
- **Estimate the tempo (BPM)**
- **See beat positions on waveform**
- **View a spectrogram**
- **Play the audio**
- **Download beat timestamps**
""")

uploaded_file = st.file_uploader("🎧 Upload your audio file", type=["wav", "mp3"])

if uploaded_file:
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Display audio player
    st.audio(uploaded_file, format="audio/wav")

    # Load audio
    y, sr = librosa.load(audio_path)

    # Beat detection
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        st.success(f"✅ Estimated Tempo: **{float(tempo):.2f} BPM**")

        # Waveform + beat markers
        st.subheader("📈 Waveform with Beat Markers")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax1)
        ax1.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
        ax1.set(title="Waveform")
        ax1.legend()
        st.pyplot(fig1)

        # Spectrogram
        st.subheader("🌈 Spectrogram")
        stft = np.abs(librosa.stft(y))
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                       sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set(title="Log-frequency Spectrogram")
        st.pyplot(fig2)

        # Export beat timestamps
        st.subheader("📤 Download Beat Times")
        beat_output = "\n".join([f"{t:.3f} sec" for t in beat_times])
        st.download_button(
            label="Download beat timestamps as .txt",
            data=beat_output,
            file_name="beat_timestamps.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error("❌ Beat detection failed. Try a different audio clip.")
        st.exception(e)
