import os
import datetime
import pyttsx3
import sounddevice as sd
import soundfile as sf
import numpy as np
import keras.models
import librosa
import speech_recognition as sr
import transformers
import threading
import sys
from utils import wake_up, action_time
from record_analysis import record_and_analyze_user_input
from text_to_speech import text_to_speech

# Load the saved emotion identification model
loaded_model = keras.models.load_model('C:/Users/dell/Desktop/AI_Virtual_C/my_model.h5')
classLabels = ('Angry', 'Fear', 'Disgust', 'Happy', 'Sad', 'Surprised', 'Neutral')

class ChatBot:
    def __init__(self, name):
        self.text = None
        print("----- Starting up", name, "-----")
        self.name = name
        self.user_input_counter = 0
        self.user_inputs = []
        self.sentiment_analysis = []
        self.record_thread = None

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)

            self.text = "ERROR"
        try:
            self.text = recognizer.recognize_google(audio)
            print("Me  -->", self.text)
        except:
            print("Me  --> ERROR")

    def run(self):
        nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        res1 = "Hello, I am zenBot. What can I do for you?"
        text_to_speech(res1)
        while True:
            self.speech_to_text()

            if wake_up(self.text, self.name):
                res = "Hello, I am ZenBot the AI. What can I do for you?"
            elif "time" in self.text:
                res = action_time()
            elif any(i in self.text for i in ["thank", "thanks"]):
                res = np.random.choice(
                    ["You're welcome!", "Anytime!", "No problem!", "Cool!", "I'm here if you need me!", "You're welcome!"]
                )
            elif any(i in self.text for i in ["exit", "close"]):
                res = np.random.choice(
                    ["Tata", "Have a good day", "Bye", "Goodbye", "Hope to meet soon", "Peace out!"]
                )
                text_to_speech(res)
                break
            else:
                if self.text == "ERROR":
                    res = "Sorry, come again?"
                else:
                    if self.record_thread and self.record_thread.is_alive():
                        self.record_thread.join()  # Wait for the previous thread to finish if it's still running

                    # Start a new thread to run record_and_analyze_user_input()
                    self.record_thread = threading.Thread(target=record_and_analyze_user_input, args=(self,))
                    self.record_thread.start()

                    chat = nlp(transformers.Conversation(self.text), pad_token_id=50256)
                    res = str(chat)
                    res = res[res.find("bot >> ") + 7:].strip()

            text_to_speech(res)

        overall_emotion = self.sentiment_analysis
        print("Overall Emotion of the conversation:", overall_emotion)
        print("----- Closing down Dev -----")
