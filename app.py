import os
import pickle
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model and PCA model
model = pickle.load(open("model.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))

classes = {0: "01_ka", 1: "02_kha", 2: "03_ga", 3: "04_gha", 4: "05_kna", 5: "06_cha", 6: "07_chha", 7: "08_ja", 8: "09_jha", 9: "10_yna"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image
        img = cv2.imread(file_path, 0)
        img = cv2.resize(img, (32, 32))
        img = img.reshape(1, -1) / 255.0
        img_pca = pca.transform(img)

        # Predict the character
        prediction = model.predict(img_pca)[0]
        character = classes[prediction]

        return jsonify({'character': character})

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
