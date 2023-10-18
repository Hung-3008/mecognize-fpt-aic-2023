from paddleocr import PaddleOCR,draw_ocr
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from paddleocr.ppstructure.kie.predict_kie_token_ser import SerPredictor, main
from models.text_correction.tool.predictor import Predictor
import time
from models.text_correction.tool.utils import extract_phrases
from utils import text_to_json, correct_text, extract_phrases, get_brand_and_related, Args


from flask import Flask, jsonify, request
import base64
import os
import numpy as np
import json
from fastapi import FastAPI
import uvicorn

import sys
sys.path.append("models/VNSpellCorrection")

from models.VNSpellCorrection.params import *
from models.VNSpellCorrection.dataset.vocab import Vocab
from models.VNSpellCorrection.models.corrector import Corrector
from models.VNSpellCorrection.models.model import ModelWrapper
from models.VNSpellCorrection.models.util import load_weights
import torch.nn.functional as F
import torch
import numpy as np




# Create an instance of FastAPI
app = FastAPI()


model_name = "tfmwtr"
dataset = "binhvq"
vocab_path = f'models/VNSpellCorrection/data/{dataset}/{dataset}.vocab.pkl'
weight_path = f'models/VNSpellCorrection/data/checkpoints/tfmwtr/{dataset}.weights.pth'

vocab = Vocab("vi")
vocab.load_vocab_dict(vocab_path)

model_wrapper = ModelWrapper(f"{model_name}", vocab)
corrector = Corrector(model_wrapper)
load_weights(corrector.model, weight_path)

args = Args()
ser_predictor = SerPredictor(args)

def inference(image, ser_predictor, corrector):

    ser_res, _, elapse = ser_predictor(image)
    diagnoes, medicines_list, date = get_brand_and_related(ser_res[0])
    
    for i in range(len(diagnoes)):
        diagnoes[i] = corrector.correct_transfomer_with_tr(diagnoes[i], num_beams=1)['predict_text']
    for i in range(len(date)):
        date[i] = corrector.correct_transfomer_with_tr(date[i], num_beams=1)['predict_text']
    for medicine in medicines_list:
        for i in range(len(medicine['quantity'])):
            medicine['quantity'][i] = corrector.correct_transfomer_with_tr(medicine['quantity'][i], num_beams=1)['predict_text']
        for i in range(len(medicine['usage'])):
            medicine['usage'][i] = corrector.correct_transfomer_with_tr(medicine['usage'][i], num_beams=1)['predict_text']
    

    return text_to_json(diagnoes, medicines_list, date)

# Define a route
@app.get("/")
def read_root():
    return "WELCOME TO MEDICAL OCR API, PLEASE USE /predict TO PREDICT"

@app.post("/predict")
async def predict(image_file: dict):
    try:
        #image_file = request.get_json(force=True)
        image_file = image_file['image']
        decoded_data = base64.b64decode(image_file)
        image_array = np.frombuffer(decoded_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # height, width = image.shape[:2]
        # aspect_ratio = float(512) / height
        # target_width = int(width * aspect_ratio)

        # image = cv2.resize(image, (target_width, 512))
        
        if image.shape[0] < 512:
            #return errors['IMAGE_TOO_SMALL']
            response = {
                'date': '',
                'medicines':[],
                'diagnose':'',
                'status': 704
            }
            return response
        # System run
        response = inference(image, ser_predictor, corrector)
        return response


    except:
        #jsonify(errors['NO_IMAGE_FOUND'])
        response = {
            'date': '',
            'medicines':[],
            'diagnose':'',
            'status': 703
        }
        return response
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

