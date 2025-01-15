import tensorflow as tf
import os
import numpy as np

class Predictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print("Model input shape:", self.model.input_shape)

    def predict(self, features: list) -> list:
        # Convert the list to a numpy array with the correct shape
        features = np.array(features).reshape(1, -1)  # Converts features to shape (1, 5)
        
        # Perform prediction
        prediction = self.model.predict(features)
        
        # Return the first prediction (in case of batch processing)
        return prediction[0].tolist()