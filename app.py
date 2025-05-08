from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the model
model = load_model('D:\yamuna\Skill4_CNN\model\CNN_waste _classification.keras')

# Preprocessing function
def preprocess_image(image):
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_data = None
    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file.filename != "":
                npimg = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                processed_image = preprocess_image(image)

                predictions = model.predict(processed_image)
                class_labels = ['Organic Waste', 'Recyclable Waste']
                predicted_class = class_labels[np.argmax(predictions)]
                prediction = predicted_class

                # Convert image to base64 to display
                import base64
                import io
                _, buffer = cv2.imencode('.jpg', image)
                image_data = base64.b64encode(buffer).decode('utf-8')

    return render_template("index.html", prediction=prediction, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
