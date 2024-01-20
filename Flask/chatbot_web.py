from voice_assistant import ChatBot
from flask import Flask, render_template, redirect
# from flask import Flask, render_template, redirect

# from voice_assistant import ChatBot

app = Flask(__name__)
ai = ChatBot(name="AIC")
overall_emotion = []


@app.route("/")
def index():
    return render_template("index.html", emotion_percentages={})




#@app.route("/process")
#def process():
#    ai.run()
#    global overall_emotion
#    overall_emotion = ai.sentiment_analysis
#    return render_template("index.html", overall_emotion=overall_emotion)
@app.route("/process")
def process():
    ai.run()
    sentiment_analysis = ai.sentiment_analysis
    emotion_percentages = calculate_emotion_percentages(sentiment_analysis)
    stress_level = calculate_stress_level(sentiment_analysis)
    return render_template("index.html", emotion_percentages=emotion_percentages, stress_level=stress_level)

def calculate_emotion_percentages(sentiment_analysis):
    total_emotions = len(sentiment_analysis)
    emotion_count = {}
    for emotion in sentiment_analysis:
        emotion_count[emotion] = emotion_count.get(emotion, 0) + 1
    emotion_percentages = {}
    for emotion, count in emotion_count.items():
        percentage = (count / total_emotions) * 100
        emotion_percentages[emotion] = round(percentage, 2)
    return emotion_percentages

def calculate_stress_level(sentiment_analysis):
    # Determine stress level based on detected emotions
    if any(emotion in sentiment_analysis for emotion in ["Angry", "Fear", "Sad", "Disgust"]):
        return "High Stress"
    elif any(emotion in sentiment_analysis for emotion in ["Happy", "Suprised"]):
        return "Low Stress"
    else:
        return "Netural Stress"

@app.route("/conversation", methods=["POST"])
def conversation():
    # Existing code for the conversation logic

    # Redirect back to the home page
    return redirect("/")

@app.route("/reset", methods=["POST"])
def reset():
    ai.sentiment_analysis = []  # Clear the emotions list
    return redirect("/")



if __name__ == "__main__":
    app.run()





