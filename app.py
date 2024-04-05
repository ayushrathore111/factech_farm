import numpy as np
import joblib
import os 
import sys
from keras.models import load_model
import keras.utils as image
from flask import Flask, redirect, url_for, request, render_template,session,flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import datetime
#loading models


knn = joblib.load('./static/knn.joblib')

preprocessor = joblib.load('./static/pre.joblib')

#flask app
app = Flask(__name__)
app.secret_key='factechhackohollics'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/factech'
db = SQLAlchemy(app)

class User(db.Model):
    email = db.Column(db.String(100),primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    yields = db.relationship('Yield', backref='user', lazy=True)
    diseases = db.relationship('Disease', backref='user', lazy=True)

    def __init__(self, username, password,email):
        self.username = username
        self.password = password
        self.email= email

class Yield(db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey('user.username'), nullable=False)
    state = db.Column(db.String(100),nullable=False)
    district = db.Column(db.String(100),nullable=False)
    area = db.Column(db.String(100),nullable=False)
    season = db.Column(db.String(100),nullable=False)
    start = db.Column(db.String(100),nullable=False)
    end = db.Column(db.String(100),nullable=False)
    yild = db.Column(db.String(100),primary_key=True,nullable=False)
    date = db.Column(db.String(100))

class Disease(db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey('user.username'), nullable=False)
    link = db.Column(db.String(100),primary_key=True)
    disease = db.Column(db.String(100))
    date = db.Column(db.String(100))


@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email, password=password).first()
        if user is not None:
            
            session['logged_in'] = True
            user = User.query.filter_by(email=email).first()
            username = user.username
            session['username']=username
            flash(f"{username}", 'success')  
            return redirect(url_for('afterlogin'))  
        else:
            error_message = "Incorrect username or password"
            flash(error_message, 'error')  
            return redirect(url_for('login'))
        
@app.route('/register/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            email = request.form['email']
            username = request.form['username']
            password = request.form['password']
            # Check if user already exists
            if User.query.filter_by(email=email).first() is not None:
                raise ValueError("User with this email already exists")
            if User.query.filter_by(username=username).first() is not None:
                raise ValueError("Username is already taken")
            
            # Add new user to the database
            db.session.add(User(email=email, username=username, password=password))
            db.session.commit()
            session['logged_in'] = True
            session['username'] = username
            flash(f"{username}", 'success')
            return redirect(url_for('afterlogin'))
        
        except ValueError as e:
            flash(str(e), 'error')
            return render_template('signUp.html')
    else:
        return render_template('signUp.html')
    
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()  # Clear the session completely
    return redirect(url_for('index'))

@app.route('/profile')
def profile():
    if 'logged_in' in session:
        username = session['username']
        user = User.query.filter_by(username=username).first()
        if user:
            yields = user.yields
            diseases = user.diseases
            return render_template('profile.html', user=user, yields=yields, diseases=diseases)
    return redirect(url_for('login'))


@app.route('/afterlogin')
def afterlogin():
    return render_template('afterlogin.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/potato')
def potato():
    return render_template('yeild.html')

@app.route('/recomend')
def recomend():
    return render_template('prediction.html')

@app.route('/detect')
def detect():
    return render_template('disease.html')

rec_model= joblib.load("./static/etr_npk1.joblib")
@app.route("/recomended",methods=['POST','GET'])
def recomended():
    if request.method=='POST':
        n = request.form['n']
        p = request.form['p']
        k = request.form['k']
        t = request.form['t']
        h = request.form['h']
        ph = request.form['ph']
        r = request.form['r']
        features= np.array([[n,p,k,t,h,ph,r]],dtype=object)
        predictions = rec_model.predict(features)
        predictions= round(predictions[0])
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if predictions in crop_dict:
            crop = crop_dict[predictions]
            return render_template("prediction.html",prediction='{} is a best crop to be cultivated'.format(crop))
        
    return render_template('prediction.html',prediction="parameters can't be left empty")
        
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
        if session.get('logged_in'):
            db.session.add(Yield(user_id=session['username'],start=start,end=end, area=area,season=season,state=state,district=district,yild=yield_value,date=datetime.datetime.now())) # type: ignore
            db.session.commit()
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
        if session.get('logged_in'):
            db.session.add(Disease(user_id=session['username'],link=img,disease=cl,date=datetime.datetime.now())) # type: ignore
            db.session.commit()
        return render_template('disease.html',cla=cl,conf=con,filename=img)

if __name__=="__main__":
    app.run(debug=False,port=5001)
    
    