from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Define class labels (update based on your dataset)
class_labels = ["No Tumor", "Tumor"]  # Update if needed

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to preprocess image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(150, 150))  # Resize
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Preprocess and predict
        image = preprocess_image(file_path)
        prediction = model.predict(image)
        predicted_class = class_labels[np.argmax(prediction)]

        return render_template("index.html", prediction=predicted_class, file_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
