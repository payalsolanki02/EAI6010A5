
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Load model in Keras 3 format
model = tf.keras.models.load_model("mnist_model.keras", compile=False)

app = FastAPI()

class InputData(BaseModel):
    pixels: list

@app.post("/predict")
def predict(data: InputData):
    try:
        img = np.array(data.pixels)

        # Reshape if it's a flat 784 list
        if img.shape == (784,):
            img = img.reshape(28, 28)

        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28)

        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return {"prediction": predicted_class, "confidence": round(confidence, 4)}
    except Exception as e:
        return {"error": str(e)}
