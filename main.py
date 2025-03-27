
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model("mnist_model.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28)

        prediction = model.predict(image)
        digit = int(np.argmax(prediction))

        return {"digit": digit}
    except Exception as e:
        return {"error": str(e)}
