import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile

# -- Page Settings --
st.set_page_config(page_title="🎧 Beat Detection", layout="centered")

# -- Custom CSS for Style --
st.markdown("""
<style>
body {
    background-color: #0f0f0f;
    color: #eeeeee;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #00ffe0;
    text-align: center;
    text-shadow: 0 0 10px #00ffe0;
}
hr {
    border: 1px solid #444;
}
.container {
    background-color: #1a1a1a;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 15px rgba(0,255,200,0.15);
}
.card {
    background-color: #222;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.upload-label {
    font-size: 18px;
    color: #00ffc6;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -- Header --
st.markdown("<h1>🎧 Beat Detection & Tempo Estimation</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -- Upload Card --
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p class='upload-label'>🔊 Upload Your Audio File (.wav or .mp3)</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["wav", "mp3"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎵 Audio Preview")
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Load audio
    y_full, sr = librosa.load(audio_path)

    # Apply harmonic-percussive separation
    y_harm, y_perc = librosa.effects.hpss(y_full)

    # Use only the percussive signal for beat detection
    y = y_perc
    y, _ = librosa.effects.trim(y)

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        st.markdown(f"<h3>✅ Estimated Tempo: {float(tempo):.2f} BPM</h3>", unsafe_allow_html=True)

        # --- Filter beat markers (min 0.3s apart) ---
        filtered_beat_times = [beat_times[0]]
        for t in beat_times[1:]:
            if t - filtered_beat_times[-1] > 0.3:
                filtered_beat_times.append(t)
        
        # --- Waveform Plot with DAW-style Beat Markers ---
        st.markdown("### 📈 Waveform with DAW-style Beat Markers")
        import matplotlib.patches as patches

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax1)

        for i in range(0, len(filtered_beat_times), 4):  # Phrase-like grouping
            t = filtered_beat_times[i]

            # Dotted vertical line
            ax1.vlines(t, -1.1, 1.1, color='#D9534F', linestyle='dotted', linewidth=1.5)

            # Triangle beat marker at top
            triangle = patches.RegularPolygon((t, 1.15), numVertices=3, radius=0.03, orientation=np.pi,
                                              color='#D9534F', zorder=5)
            ax1.add_patch(triangle)

            # Optional rectangle block at the bottom
            ax1.add_patch(patches.Rectangle((t - 0.05, -1.2), 0.1, 0.08, color='#D9534F', alpha=0.5))

        ax1.set(title="Waveform")
        ax1.set_ylim([-1.3, 1.2])
        ax1.legend(["Beat Markers"], loc="upper right")
        st.pyplot(fig1)


        # Spectrogram
        st.markdown("### 🌈 Spectrogram")
        stft = np.abs(librosa.stft(y))
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                       sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set(title="Log-frequency Spectrogram")
        st.pyplot(fig2)

        # Download Timestamps
        st.markdown("### 📤 Export Beat Times")
        beat_output = "\n".join([f"{t:.3f} sec" for t in beat_times])
        st.download_button(
            label="📥 Download beat timestamps as .txt",
            data=beat_output,
            file_name="beat_timestamps.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error("❌ Beat detection failed. Please try another file.")
        st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<p style='text-align:center; color: gray;'>🕒 Waiting for file upload...</p>", unsafe_allow_html=True)
