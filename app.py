import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os
import torch
import soundfile as sf
from datasets import load_dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

def main():
    # Choose image source
    image_source = st.radio("Select Image Source:", ("Upload Image", "Open from URL"))

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
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                image = Image.open(response.raw)
            else:
                st.error("Error loading image from URL.")
                image = None
        else:
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
        output_caption = generate_caption(caption_processor, caption_model, image)

        # Generate speech from the caption
        generate_speech(speech_processor, speech_model, speech_vocoder, speaker_embeddings, output_caption)

        # Play the generated sound
        play_sound()

        # Display the caption
        st.subheader("Caption:")
        st.write(output_caption)

if __name__ == "__main__":
    main()