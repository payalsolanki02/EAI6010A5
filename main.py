
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
import tensorflow as tf
from model import load_mnist_model

app = FastAPI()
model = load_mnist_model()

@app.get("/")
def home():
    return HTMLResponse(content=open("index.html").read(), status_code=200)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L").resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return {"class": predicted_class, "confidence": confidence}
