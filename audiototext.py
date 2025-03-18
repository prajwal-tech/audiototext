import torch
import torchaudio
import pandas as pd
import streamlit as st
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load the pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Function to transcribe audio
def transcribe_audio(file_path):
    # Load audio file
    speech_array, sampling_rate = torchaudio.load(file_path)

    # Resample if necessary
    target_sampling_rate = 16000
    if sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sampling_rate)
        speech_array = resampler(speech_array)

    # Convert tensor to numpy array
    speech = speech_array.squeeze().numpy()

    # Preprocess the audio file
    input_values = processor(speech, return_tensors="pt", sampling_rate=target_sampling_rate).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode predicted text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

# Streamlit UI
st.title("🎙️ Audio to Text Converter")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, FLAC)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Save file locally
    file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Transcribe audio
    st.success("File uploaded successfully! Processing...")
    transcription = transcribe_audio(file_path)

    # Show transcription
    st.subheader("📝 Transcribed Text")
    st.write(transcription)

    # Save to CSV
    df = pd.DataFrame([[transcription]], columns=["Text Data"])
    csv_filename = "TextData.csv"
    df.to_csv(csv_filename, index=False)

    # Provide download link
    st.download_button(label="📥 Download CSV", data=df.to_csv(index=False), file_name=csv_filename, mime="text/csv")
