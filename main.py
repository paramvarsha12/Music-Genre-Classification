import streamlit as st
import tensorflow as tf
import pickle
from utils import audio_to_melspectrogram_image, load_image
import os
import tempfile

@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model('models/music_genre_cnn.h5')
    with open('models/label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, label_encoder = load_model_and_encoder()

st.set_page_config(page_title="Music Genre Classifier", layout="centered")
st.title("Music Genre Classifier")
st.markdown("Upload a `.wav` file and I’ll predict the genre of the music.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')
    audio_to_melspectrogram_image(temp_path, save_path='temp.png')
    st.image('temp.png', caption='Mel Spectrogram', use_container_width=True)


    if st.button("Predict Genre"):
        img = load_image('temp.png')
        prediction = model.predict(img)
        predicted_index = prediction.argmax()
        predicted_genre = label_encoder.inverse_transform([predicted_index])[0]
        st.success(f"Predicted Genre: {predicted_genre.capitalize()}")
