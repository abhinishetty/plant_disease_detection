import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Define labels with additional information
labels = {
    0: "Healthy: The plant is in good condition with no visible disease symptoms.",
    1: "Powdery Mildew: A fungal disease that creates white powdery spots on leaves. Prompt fungicide treatment is recommended.",
    2: "Rust: A fungal infection causing orange, yellow, or brown spots on leaves. Remove infected leaves and consider fungicide application."

}

def getResult(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)

    # Make predictions
    predictions = model.predict(x)[0]
    predicted_label = np.argmax(predictions)

    # Fetch detailed description from labels
    conclusion = labels[predicted_label]
    return conclusion


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded file
        f = request.files['file']

        # Save the file to the uploads directory
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Generate prediction and conclusion
        conclusion = getResult(file_path)
        return conclusion  # Return the detailed conclusion
    return None


if __name__ == '__main__':
    app.run(debug=True)
