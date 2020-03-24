import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)
dir_path = 'D:/SOFT_COPIES/flask_app_for_running_webpage/face_recon/flask-app'
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

graph = tf.get_default_graph()
with graph.as_default():
    # load model at very first
    model = load_model('D://FaceR/BASHPATH/model12.h5')
    # model = load_model(STATIC_FOLDER + '/' + 'model12.h5')



# call model to predict an image
def api(full_path):
    img = cv2.imread(full_path)
    file = cv2.resize(img, (48, 48))
    file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    file = file.reshape((-1, 48, 48,1))
    data = file * 1.0 / 255

    with graph.as_default():
        predicted = model.predict(data)
        return predicted


# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust',4:'fear',5:'happy',6:'sadness',7:'surprise'}
        # indices = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy',4:'Sad',5:'Suprise',6:'Neitral'}

        result = api(full_name)

        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]

    return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True
