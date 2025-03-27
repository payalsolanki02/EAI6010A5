
from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
from keras.layers import TFSMLayer

# Load model from exported SavedModel folder
model = TFSMLayer("mnist_model", call_endpoint="serving_default")

app = FastAPI()

class InputData(BaseModel):
    pixels: list  # 784 or 28x28 pixel values

@app.post("/predict")
async def predict(data: InputData):
    try:
        img = np.array(data.pixels)

        # Reshape if flat
        if img.shape == (784,):
            img = img.reshape(28, 28)

        # Normalize and reshape for batch
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28)

        prediction = model(img, training=False).numpy()
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {"prediction": predicted_class, "confidence": round(confidence, 4)}
    except Exception as e:
        return {"error": str(e)}
