from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
import tensorflow as tf

from model import preprocess_image

app = FastAPI()

model = tf.keras.models.load_model("mnist_model.keras")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L")
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)
    predicted_class = int(np.argmax(prediction))
    return {"prediction": predicted_class}