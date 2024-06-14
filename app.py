import joblib
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import os

from main import *

app = Flask(__name__)





IMAGE_SIZE = (224, 224)

loaded_model = models.load_model('epoch50.h5', custom_objects={'KerasLayer': hub.KerasLayer})
svm_model = joblib.load('svm_model.pkl')

def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

@app.route('/')
def index():
    return render_template("upload.html")



a=[]
@app.route('/upload_image', methods = ['POST', 'GET'])
def upload_image():
       if request.method == 'POST':
           f = request.files['file']
           basepath = os.path.dirname(__file__)
           file_path = os.path.join(basepath,"static/uploads/original.jpg")
           f.save("static/uploads/original.jpg")
           class_labels = ['both leaf ', 'healthy  leaf', 'rust  leaf', 'scab leaf ']
           input_image_path = 'static/uploads/original.jpg'
           input_image = preprocess_image(input_image_path, IMAGE_SIZE)  
           input_features = loaded_model.predict(input_image)
           input_features_flattened = np.reshape(input_features, (input_features.shape[0], -1))
           svm_prediction = svm_model.predict(input_features_flattened)
           predicted_class = np.argmax(input_features, axis=1)
           a.append(predicted_class[0])
           print(predicted_class[0])
           predicted_label = class_labels[predicted_class[0]]
           print(type(predicted_label))
           confidence_score = input_features[0][predicted_class][0]
           if predicted_class[0]==1:
                return "Cannot segment because this is a healthy leaf"
           else:
                segment=deepdream_seg()
                return render_template("results_chest.html",result=predicted_label)
            
       return render_template("upload.html")


@app.route('/segment', methods = ['POST', 'GET'])
def segment():
    if a[-1]==1:
        return "cannot segment this leaf because this is healthy leaf"
    else:
        return render_template("final.html")

        
        

if __name__ == '__main__':
   app.run(port="3200", debug=False)
