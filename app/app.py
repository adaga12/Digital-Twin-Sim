# HTTP SERVER

import json

from flask import Flask, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from simulator import Simulator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from store import QRangeStore


class Base(DeclarativeBase):
    pass


############################## Application Configuration ##############################

app = Flask(__name__)
CORS(app, origins=["http://localhost:3030"])

db = SQLAlchemy(model_class=Base)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
db.init_app(app)


############################## Database Models ##############################


class Simulation(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[str]


with app.app_context():
    db.create_all()


############################## API Endpoints ##############################


@app.get("/")
def health():
    return "<p>Sedaro Nano API - running!</p>"


@app.get("/simulation")
def get_data():
    simulation = Simulation.query.order_by(Simulation.id.desc()).first()
    return simulation.data if simulation else []


@app.post("/simulation")
def simulate():
    init = request.json
    for key in init.keys():
        init[key]["time"] = 0
        init[key]["timeStep"] = 0.01

    # Initialize simulation with updated model
    model_path = "state_predictor_nn.keras"
    store = QRangeStore()
    simulator = Simulator(store=store, init=init, model_path=model_path)

    # Run the simulation
    simulator.simulate()

    # Save results to the database
    simulation_data = json.dumps(store.store)
    simulation = Simulation(data=simulation_data)
    db.session.add(simulation)
    db.session.commit()

    return store.store


if __name__ == '__main__':
    app.run(debug=True)