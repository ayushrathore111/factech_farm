import numpy as np
import joblib
import os 
import sys
from keras.models import load_model
import keras.utils as image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#loading models


knn = joblib.load('./static/knn.joblib')

preprocessor = joblib.load('./static/pre.joblib')

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/potato')
def potato():
    return render_template('yeild.html')

@app.route('/detect')
def detect():
    return render_template('disease.html')


@app.route("/potato-predicted",methods=['POST'])
def potatopredict():
    if request.method == 'POST':
        start = request.form['start']
        end = request.form['end']
        area= request.form['area']
        season = request.form['season']
        state = request.form['state']
        district  = request.form['district']
        features = np.array([[state,district,season,start,end,area]],dtype=object)
        print(features)
        transformed_features = preprocessor.transform(features)
        prediction = knn.predict(transformed_features)
        
        prediction_value = float(prediction[0])
        area_value = float(area)
        yield_value = prediction_value / area_value
        yield_value=round(yield_value, 2)
        return render_template('yeild.html',prediction = 'The Predicted Production is {}tonnes and Yield is {}tonnes/hectare'.format(prediction[0],yield_value))
@app.route("/pigeon-predicted",methods=['POST'])
def pigeonpredict():
    if request.method == 'POST':
        start = request.form['start']
        end = request.form['end']
        area= request.form['area']
        season = request.form['season']
        state = request.form['state']
        district  = request.form['district']
        features = np.array([[state,district,season,start,end,area]],dtype=object)
        print(features)
        transformed_features = preprocessor.transform(features)
        prediction = knn.predict(transformed_features)
        
        prediction_value = float(prediction[0])
        area_value = float(area)
        yield_value = prediction_value / area_value
        yield_value=round(yield_value, 2)
        return render_template('pigeon.html',prediction = 'The Predicted Production is {}tonnes and Yield is {}tonnes/hectare'.format(prediction[0],yield_value))


# Load your trained model
model = load_model('./static/potatoes.h5')
CLASS_NAMES = ['Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy']
model.make_predict_function()  
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))  # Resize image to match model's expected sizing
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the batch size used by the model
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the class of the uploaded image
def predict_image(image_path):
    img_array = preprocess_image(image_path)  # Preprocess the image
    prediction = model.predict(img_array)  # Predict the class probabilities
    predicted_class = np.argmax(prediction, axis=1)[0] 
    # Get the index of the class with the highest probability
    predicted_class=CLASS_NAMES[predicted_class]
    confidence = np.max(prediction[0])
    return predicted_class,confidence

@app.route('/disease-predicted', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)
        cl,con= predict_image(file_path)
        img=url_for('static', filename='uploads/' + secure_filename(f.filename))
        return render_template('disease.html',cla=cl,conf=con,filename=img)

if __name__=="__main__":
    app.run(debug=False,port=5001)
    
    