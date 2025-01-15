import tensorflow as tf
from tensorflow import keras
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from keras.api.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import os
from simulator import Simulator
from store import QRangeStore
import numpy as np

# Define model path and initial states
model_path = "state_predictor_nn.keras"

# Define the initial state of the simulation, required to start the simulator
init = {
    "Body1": {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "vx": 1.0, "vy": 1.0, "vz": 1.0,
        "mass": 5.0, "time": 0.0, "timeStep": 0.01
    },
    "Body2": {
        "x": 10.0, "y": 10.0, "z": 10.0,
        "vx": -1.0, "vy": -1.0, "vz": -1.0,
        "mass": 10.0, "time": 0.0, "timeStep": 0.01
    }
}

# Create store and initialize simulator
store = QRangeStore()
simulator = Simulator(store=store, init=init)

# Run simulation with default iterations
simulator.simulate(iterations=500)

# Get training data
X_train, y_train = simulator.get_training_data()
print("Training features shape:", X_train.shape)
print("Training labels shape:", y_train.shape)

if len(X_train) < 10: # Make sure enough training data is collected
    raise ValueError(f"Not enough training data collected ({len(X_train)} samples). Need at least 10 samples.")

# Ensure data is properly formatted
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

# Define model using Sequential API
model = Sequential([
    Dense(128, activation='relu', input_shape=(5,)),
    BatchNormalization(),
    Dropout(0.2), #Enforces Dropout techniques to prevent overfitting
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(4)  # Output layer
])

# Learning rate scheduler function
def lr_schedule(epoch, lr):
    if epoch > 50:
        return lr * 0.1
    return lr

# Compile the model suited for regression tasks
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error as another metric
)

# TensorBoard
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=20,  # Stop after 20 epochs if there is no improvement
    restore_best_weights=True  # Restore the best observed weights
)

# Train the model
model.fit(
    X_train, y_train, 
    validation_split=0.2,  # Use 20% of the data for validation
    epochs=100, 
    batch_size=8,  # Use a mini-batch size of 8, can be changed around
    callbacks=[LearningRateScheduler(lr_schedule), tensorboard_callback, early_stopping]
)

# Save the trained model
model.save(model_path)  # Save the model as state_predictor_nn.keras
print("Model saved as 'state_predictor_nn.keras'")

# Load the trained model for future predictions
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")

def example_predictor(model, new_input_data):
    """
    Predict the next state using the trained model.

    :param model: The trained Keras model.
    :param new_input_data: A list or numpy array of input features.
    :return: Predicted next state (x, y, vx, vy).
    """
    # Ensure the input data is in the correct shape
    input_data = np.array([new_input_data], dtype=np.float32)
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Output the predicted state
    predicted_state = {
        "x": prediction[0][0],
        "y": prediction[0][1],
        "vx": prediction[0][2],
        "vy": prediction[0][3]
    }
    
    return predicted_state


# Example input data
example_input = [5.0, 5.0, 0.5, 0.5, 1.0]

# Use the trained model to predict the next state
predicted_state = example_predictor(model, example_input)

print("Predicted next state:", predicted_state)
