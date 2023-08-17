import speech_recognition as sr
import pyttsx3
import pyaudio
import webbrowser
import datetime
import pywhatkit
import os
import wave

def transform():
    r = sr.Recognizer() 
    with sr.Microphone() as source:
        #print("Listening..")
        audio = r.listen(source)
        q = ""
        try:
            q = r.recognize_google(audio, language="id")
            print("I'm listening: " + q)
        except sr.UnknownValueError:
            print("Sorry I didn't understand")
        except sr.RequestError as e:
            print("Request Failed: {0}".format(e))
    return q

if __name__=="__main__":
    transform()

