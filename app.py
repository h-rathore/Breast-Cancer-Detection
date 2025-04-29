from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("model.h5")
IMG_SIZE = 128
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            prediction = model.predict(img)
            class_index = np.argmax(prediction)

            labels = ["Benign", "Malignant", "Normal"]
            result = labels[class_index]

            return render_template("result.html", result=result, filename=filename)
    return render_template("index.html")

@app.route("/display/<filename>")
def display_image(filename):
    return f"/static/uploads/{filename}"

if __name__ == "__main__":
    app.run(debug=True)
