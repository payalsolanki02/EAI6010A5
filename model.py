
import tensorflow as tf

def load_mnist_model():
    model = tf.keras.models.load_model("mnist_model.keras")
    return model
