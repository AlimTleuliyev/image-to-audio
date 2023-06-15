# Image Captioning and Text-to-Speech Application

This project is an Image Captioning and Text-to-Speech application that generates descriptive captions for uploaded images and converts the captions into speech. It utilizes state-of-the-art models for image captioning and text-to-speech synthesis, providing a seamless user experience for visually impaired individuals, content creators, or anyone interested in exploring multimodal AI applications.

## Features

- **Image Captioning**: The application employs [Salesforce's BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) image captioning model, which has been trained on large-scale image-caption datasets. It generates accurate and contextually relevant captions for uploaded images, allowing users to understand the content of the image without relying solely on visual perception.

- **Text-to-Speech Synthesis**: [Microsoft's SpeechT5](https://huggingface.co/microsoft/speecht5_tts) model is employed for text-to-speech synthesis, converting the generated captions into natural-sounding speech. The SpeechT5 model incorporates advanced techniques for speech generation, producing high-quality and expressive speech output.

- **Multiple Input Options**: The application supports multiple input options for convenience. Users can either upload an image from their local device or provide the URL of an image hosted online. This flexibility allows users to easily access and process images from various sources.

- **Real-time Processing**: The image captioning and text-to-speech synthesis are performed in real-time, ensuring quick and responsive results. Users can instantly see the generated caption and hear the corresponding speech output, enabling a seamless and interactive experience.

- **User-friendly Interface**: The application features a user-friendly interface built using the Streamlit framework. It provides clear instructions, intuitive image upload options, and visually appealing visualizations. Users can easily interact with the application, making it accessible to individuals with varying technical backgrounds.

## Installation

To run the Image Captioning and Text-to-Speech application locally, follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/your-repo.git
   ```
2. Install the required dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```
3. Run the application:
   ```shell
   streamlit run app.py
   ```
   The application will be accessible in your web browser at http://localhost:8501.

## Contributions and Support

Contributions, bug reports, and feature requests are welcome! If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository. You can also reach out to the project maintainer, Alim Tleuliyev, at alim.tleuliyev@nu.edu.kz for further assistance or inquiries.
   
