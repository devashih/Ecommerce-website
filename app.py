import os
import tensorflow as tf
import warnings
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from io import BytesIO

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = Flask(__name__)
cors = CORS(app)

global MODEL
global CLASSES

# DigiLocker API credentials
DIGILOCKER_CLIENT_ID = 'your_client_id'
DIGILOCKER_CLIENT_SECRET = 'your_client_secret'
DIGILOCKER_REDIRECT_URI = 'your_redirect_uri'

@app.route("/", methods=["GET"])
def default():
    return jsonify({"Hello I am Chitti": "Speed 1 Terra Hertz, Memory 1 Zeta Byte"})

@app.route("/predict", methods=["GET"])
def predict():
    src = request.args.get("src")
    response = requests.get(src)
    try:
        image = Image.open(BytesIO(response.content))
        image = image.resize((128, 128))
        image = np.array(image)
        image = image.astype("float") / 255.0
        image = np.expand_dims(image, axis=0)
        pred = MODEL.predict(image)
        return jsonify({"class": CLASSES[int(np.argmax(pred, axis=1))]})
    except Exception as e:
        return jsonify({"Uh oh": str(e)})

@app.route("/digilocker/auth", methods=["GET"])
def digilocker_auth():
    # Redirect user to DigiLocker for authentication
    auth_url = f"https://api.digilocker.gov.in/v1/auth?client_id={DIGILOCKER_CLIENT_ID}&redirect_uri={DIGILOCKER_REDIRECT_URI}&response_type=code"
    return jsonify({"auth_url": auth_url})

@app.route("/digilocker/callback", methods=["GET"])
def digilocker_callback():
    # Handle the callback from DigiLocker
    code = request.args.get("code")
    token_url = "https://api.digilocker.gov.in/v1/token"
    payload = {
        'client_id': DIGILOCKER_CLIENT_ID,
        'client_secret': DIGILOCKER_CLIENT_SECRET,
        'code': code,
        'redirect_uri': DIGILOCKER_REDIRECT_URI,
        'grant_type': 'authorization_code'
    }
    response = requests.post(token_url, data=payload)
    token_data = response.json()
    access_token = token_data.get("access_token")
    
    # Now you can use the access token to fetch documents
    return jsonify({"access_token": access_token})

@app.route("/digilocker/documents", methods=["GET"])
def digilocker_documents():
    access_token = request.args.get("access_token")
    documents_url = "https://api.digilocker.gov.in/v1/documents"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(documents_url, headers=headers)
    documents = response.json()
    return jsonify(documents)

if __name__ == "__main__":
    MODEL_PATH = os.path.abspath("./models/image/dump/mobile_net.h5")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    CLASSES = ["control", "gore", "pornography"]
    app.run(threaded=True, debug=True)
