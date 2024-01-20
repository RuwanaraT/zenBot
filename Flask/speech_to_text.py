import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with sr.Microphone() as mic:
        print("Listening...")
        audio = recognizer.listen(mic)

        text = "ERROR"
    try:
        text = recognizer.recognize_google(audio)
        print("Me  -->", text)
    except:
        print("Me  --> ERROR")

    return text


