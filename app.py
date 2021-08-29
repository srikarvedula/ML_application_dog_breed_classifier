from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from main import load_paths
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'dog_classifier.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 256.
    x = np.expand_dims(x, axis=0)
    output = model.predict(x)
    root_path = 'C:/Users/srika/PycharmProjects/dog_classifier/dataset/'
    _, _, label_to_class = load_paths(root_path, 'train')
    preds = [np.argmax(output)]
    cls_preds = [label_to_class[l] for l in preds]
    print(cls_preds)
    new_values = " ".join(cls_preds)
    if '_' in new_values:
        val = list(new_values.split('_'))
        new_values = " ".join(val)
        print(new_values)
        message="This dog breed is " + new_values
        return message
    else:
        print(new_values)
        message = "This dog breed is " + new_values
        return message
    # img = image.load_img(img_path, target_size=(256, 256))
    # img_arr=np.stack(np.asarray(Image.open(img).convert('RGB').resize((256, 256))), axis=0)
    # train_mean = np.mean(np.mean(np.mean(img_arr, axis=0), axis=0), axis=0)
    # img_norm = (img_arr - train_mean[None, None, None, :]) / 256.0
    # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # ## Scaling
    # x = x / 255
    # x = np.expand_dims(x, axis=0)
    #
    # # Be careful how your trained model deals with the input
    # # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)


    # output = model.predict(img_norm)
    # root_path = 'C:/Users/srika/PycharmProjects/dog_classifier/dataset/'
    # _,_,label_to_class = load_paths(root_path, 'train')
    # preds = [np.argmax(output)]
    # cls_preds = [label_to_class[l] for l in preds]
    # print(cls_preds)

    # preds = np.argmax(preds, axis=1)
    # if preds == 0:
    #     preds = "The Person is Infected With Pneumonia"
    # else:
    #     preds = "The Person is not Infected With Pneumonia"
    #
    # return cls_preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)