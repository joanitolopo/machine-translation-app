# Import Packagas
import time
import os
import cv2
from sys import float_repr_style, path
from requests.api import get

import pandas as pd
import numpy as np

import streamlit as st
from streamlit.elements import text
from streamlit.elements.arrow_vega_lite import _CHANNELS

from testmodel import prediksi
from utils import translate_sentence, tokenize_src, tokenize_trg
from text_to_speech.speak import speak
from article_translation.crawl_article import get_data
from image_to_text.bound_detect import capture
from speech_to_text.recognize import transform


def main():
    # Register your pages
    st.set_page_config(page_title="Machine Translation", page_icon="🤖")
    pages = {
        "Machine Translation": page_first,
        "Article Translation": page_second,
    }

    st.sidebar.title("Services")

    # Widget to select your page, you can choose between radio buttons or a selectbox
    page = st.sidebar.radio("Select pages:", tuple(pages.keys()))
    #page = st.sidebar.selectbox("Select your page", tuple(pages.keys()))

    # Display the selected page
    pages[page]()

def page_first():
    # Title for the page and nice icon
    # Header
    st.title("Translation")

    def mirror():
        with col2:

            # Dropdown menu to select a language pair
            lang_pair_target = st.selectbox("Pilih Bahasa", (
                "MelayuKupang", "Soppeng", "Kaili", "MelayuAmbon",
                "Makasar"))
            # st.write('You selected:', lang_pair)
        
        return lang_pair_target


    output = ''
    col1, col2 = st.columns(2)
    with col1:
        # Form to add your items
        with st.form("form_source"):

            # Dropdown menu to select a language pair
            lang_pair_source = st.selectbox("Pilih Bahasa", (
                                    "Indonesia","English"))

            # Textarea to type the source text.
            max_chars = 200
            user_input = st.text_area("Masukan Teks", max_chars=max_chars)

            target = mirror()
            key = lang_pair_source + 'to' + target
            print("Password: ", key)
        
            # Create a button
            submitted = st.form_submit_button("Translate")

            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 20px;">Feature belum didukung</p>'
            
            if target == "English":
                st.markdown(new_title, unsafe_allow_html=True)

            # If the button pressed, print the translation
            if submitted:
                output = prediksi(lang_pair_source, target, user_input, max_len=max_chars)

            
            with st.expander("Advanced Features"):
                mic = st.form_submit_button("Mic")
                cam = st.form_submit_button("Camera")

                if mic:
                    st.text("Listening..")
                    text = transform()
                    print(text)
                    output = prediksi(lang_pair_source, target, text, max_len=max_chars)
                    

                if cam:
                    text = capture()
                    print(text)
                    output = prediksi(lang_pair_source, target, text, max_len=max_chars)

        uploaded_file = st.file_uploader("Choose a picture")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            text = capture(file_bytes)
            text = " ".join(text.split())
            if st.button("Translate"):
                output = prediksi(lang_pair_source, target, text, max_len=max_chars)
            
    with col2:   
        # Textarea to type the source text.
        result = st.text_area("Terjemahan", output, max_chars=200)
        speech = st.button("Speech")
        
        path="sample.txt"
        fd = os.open(path, os.O_RDWR)
        line = str.encode(result)
        numBytes = os.write(fd, line)
        os.close(fd)
        
        if speech:
            f = open("sample.txt", "r").read()
            speak(f)
            
    st.caption('Copyright 2021. Tribelingo')
    
    


    # Optional Style
    st.markdown(""" <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .reportview-container .main .block-container{
            padding-top: 0rem;
            padding-right: 0rem;
            padding-left: 0rem;
            padding-bottom: 0rem;
        } </style> """, unsafe_allow_html=True)


    st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)

def page_second():
    st.title("Article Translation")

    link = st.text_input('Link')

    option = st.selectbox(
        'Translate to: ',
        ('MelayuKupang', 'Kaili', 'MelayuAmbon',
        'Soppeng', 'Makasar'))

    if link :
        data = get_data(link)
        for indek, article in enumerate(data):
            output = prediksi("Indonesia", option, article)
            st.write({article:output})

if __name__ == "__main__":
    main()











