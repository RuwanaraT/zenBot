import keras.models
import numpy as np

# Load the saved emotion identification model
loaded_model = keras.models.load_model('C:/Users/malithg/PycharmProjects/AI_Virtual_C/my_model.h5')
classLabels = ('Angry', 'Fear', 'Disgust', 'Happy', 'Sad', 'Surprised', 'Neutral')
