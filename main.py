from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from keras.layers import TFSMLayer

# ✅ Load exported SavedModel using TFSMLayer
model = TFSMLayer("mnist_model", call_endpoint="serving_default")

app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    pixels: list  # Should be a flattened 784-length list or a 28x28 nested list

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to NumPy array
        img = np.array(data.pixels)

        # Reshape flat input to 28x28 if needed
        if img.shape == (784,):
            img = img.reshape(28, 28)

        # Normalize and reshape for model input
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28)

        # Run prediction using TFSMLayer
        output = model(img, training=False).numpy()
        predicted_class = int(np.argmax(output))
        confidence = float(np.max(output))

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {"error": str(e)}

