
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model("mnist_model.h5", compile=False)

app = FastAPI()

class InputData(BaseModel):
    pixels: list  # 28x28 list or flat 784

@app.post("/predict")
def predict(data: InputData):
    try:
        img = np.array(data.pixels)

        # Reshape if needed
        if img.shape == (784,):
            img = img.reshape(28, 28)

        # Preprocess
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28)

        # Predict
        pred = model.predict(img)
        predicted = int(np.argmax(pred))
        confidence = float(np.max(pred))

        return {"prediction": predicted, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}
