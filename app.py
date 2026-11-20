
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load your trained CNN model
model = load_model("Fruit_Freshness_CNN.keras")

# Keep this order exactly the same as model training
class_names = ["Fresh", "Mild", "Rotten"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        if "file" not in request.files:
            return render_template("index.html", error="Please select an image.")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="Please select an image.")

        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Preprocess image
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        result = model.predict(img_array)
        predicted_index = np.argmax(result[0])

        prediction = class_names[predicted_index]
        confidence = round(float(result[0][predicted_index]) * 100, 2)

        image_path = "/" + image_path.replace("\\", "/")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)
