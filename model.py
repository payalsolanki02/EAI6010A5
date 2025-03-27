import numpy as np

def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28)
    return image