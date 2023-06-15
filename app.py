import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os
import torch
import soundfile as sf
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Model Description
model_description = """
This application utilizes image captioning and text-to-speech models to generate a caption for an uploaded image 
and convert the caption into speech.

The image captioning model is based on [Salesforce's BLIP architecture](https://huggingface.co/Salesforce/blip-image-captioning-base), which can generate descriptive captions for images.

The text-to-speech model, based on [Microsoft's SpeechT5](https://huggingface.co/microsoft/speecht5_tts), converts the generated caption into speech with the help of a 
HiFiGAN vocoder.
"""


@st.cache_resource
def initialize_image_captioning():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def initialize_speech_synthesis():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return processor, model, vocoder, speaker_embeddings

def generate_caption(processor, model, image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    output_caption = processor.decode(out[0], skip_special_tokens=True)
    return output_caption

def generate_speech(processor, model, vocoder, speaker_embeddings, caption):
    inputs = processor(text=caption, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write("speech.wav", speech.numpy(), samplerate=16000)

def play_sound():
    audio_file = open("speech.wav", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

def visualize_speech():
    data, samplerate = sf.read("speech.wav")
    duration = len(data) / samplerate

    # Create time axis
    time = np.linspace(0., duration, len(data))

     # Plot the speech waveform
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, data)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Speech Waveform")

    # Display the plot using st.pyplot()
    st.pyplot(fig)

def main():
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Alim Tleuliyev")
    st.sidebar.markdown("Contact: [alim.tleuliyev@nu.edu.kz](mailto:alim.tleuliyev@nu.edu.kz)")

    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<div class='title'>Image Captioning and Text-to-Speech</div>", unsafe_allow_html=True)

    # Model Description
    st.markdown("<div class='description'>" + model_description + "</div>", unsafe_allow_html=True)

    # Instructions
    st.markdown("<div class='title'>Instructions</div>", unsafe_allow_html=True)
    st.markdown("1. Upload an image or provide the URL of an image.")
    st.markdown("2. Click the 'Generate Caption and Speech' button.")
    st.markdown("3. The generated caption will be displayed, and the speech will start playing.")


    # Choose image source
    image_source = st.radio("Select Image Source:", ("Upload Image", "Open from URL"))

    image = None

    if image_source == "Upload Image":
        # File uploader for image
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

    else:
        # Input box for image URL
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Error loading image from URL.")
                    image = None
            except requests.exceptions.RequestException as e:
                st.error(f"Error loading image from URL: {e}")
                image = None

    # Generate caption and play sound button
    if image is not None:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Initialize image captioning models
        caption_processor, caption_model = initialize_image_captioning()

        # Initialize speech synthesis models
        speech_processor, speech_model, speech_vocoder, speaker_embeddings = initialize_speech_synthesis()

        # Generate caption
        with st.spinner("Generating Caption..."):
            output_caption = generate_caption(caption_processor, caption_model, image)

        # Display the caption
        st.subheader("Caption:")
        st.write(output_caption)
        
        # Generate speech from the caption
        with st.spinner("Generating Speech..."):
            generate_speech(speech_processor, speech_model, speech_vocoder, speaker_embeddings, output_caption)

        
        st.subheader("Audio:")
        # Play the generated sound
        play_sound()

        # Visualize the speech waveform
        with st.expander("See visualization"):
            visualize_speech()


if __name__ == "__main__":
    main()