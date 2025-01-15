# SIMULATION
from functools import reduce
from operator import __or__

from modsim import propagate
from store import QRangeStore
from model_handler import Predictor
import numpy as np


class Simulator:
    def __init__(self, store, init, model_path=None):
        self.store = store
        store[-999999999, 0] = init  # Store the initial state in the QRangeStore
        self.init = init
        self.times = {agentId: state["time"] for agentId, state in init.items()}
        self.predictor = Predictor(model_path) if model_path else None
        
        # Data containers for training
        self.features = []
        self.labels = []

    def read(self, t):
        try:
            data = self.store[t]
        except IndexError:
            data = []
        return {key: value for d in data for key, value in d.items()}

    def extract_features(self, state, other_state):
        # Return 5 features with default mass
        return [
            state["x"],              # initial x position
            state["y"],              # initial y position
            state["vx"],             # initial x velocity
            state["vy"],             # initial y velocity
            state.get("mass", 1.0)   # mass with default value of 1.0
        ]

    def extract_labels(self, state, new_state):
        # Return labels (next state: x, y, vx, vy)
        return [
            new_state["x"],  # next x position
            new_state["y"],  # next y position
            new_state["vx"], # next x velocity
            new_state["vy"]  # next y velocity
        ]

    def propagate(self, agent_id, universe):
        state = universe[agent_id]
        other_state = universe['Body1'] if agent_id == 'Body2' else universe['Body2']

        # Extract features
        features = self.extract_features(state, other_state)

        # If a model is available, predict the next state
        if self.predictor:
            features = np.array([features])  # Wrap features as a batch
            prediction = self.predictor.predict(features)

            new_state = {
                "x": prediction[0],
                "y": prediction[1],
                "vx": prediction[2],
                "vy": prediction[3],
                "time": state["time"] + 0.001
            }
        else:
            new_state = {
                "x": state["x"] + state["vx"] * 0.001,
                "y": state["y"] + state["vy"] * 0.001,
                "vx": state["vx"],
                "vy": state["vy"],
                "time": state["time"] + 0.001
            }

        # Store the features and labels
        self.features.append(features)
        self.labels.append(self.extract_labels(state, new_state))

        return new_state

    def simulate(self, iterations=100):
        print(f"Starting simulation with {iterations} iterations...")
        for i in range(iterations):
            for agent_id in self.init:
                t = self.times[agent_id]
                universe = self.read(t - 0.001)
                
                # Ensure the universe is valid
                if set(universe) == set(self.init):
                    new_state = self.propagate(agent_id, universe)

                    # Validate time range before storing
                    if new_state["time"] > t:
                        self.store[t, new_state["time"]] = {agent_id: new_state}
                        self.times[agent_id] = new_state["time"]
                    else:
                        print(f"Skipped invalid time range: {t} -> {new_state['time']}")

            if i % 100 == 0:
                print(f"Iteration {i}: Collected {len(self.features)} samples")

        print(f"Simulation complete. Total samples collected: {len(self.features)}")
        self.save_training_data()

    def save_training_data(self):
        # Save features and labels to files
        np.save('features.npy', np.array(self.features))
        np.save('labels.npy', np.array(self.labels))
        print("Training data saved to 'features.npy' and 'labels.npy'.")

    def get_training_data(self):
        return np.array(self.features), np.array(self.labels)