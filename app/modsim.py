# MODELING & SIMULATION

from random import random

import numpy as np


def propagate(agent_id, universe):
    """Propagate agentId from `time` to `time + timeStep`."""
    # Get simulation state
    self_state = universe[agent_id]
    time = self_state["time"]
    time_step = self_state["timeStep"]

    if agent_id == 'Body1':
        other_body = 'Body2'
    else:
        other_body = 'Body1'
    other_state = universe[other_body]

    # Use law of gravitation to update position and velocity
    r_self = np.array([self_state['x'], self_state['y'], self_state['z']])
    v_self = np.array([self_state['vx'], self_state['vy'], self_state['vz']])
    r_other = np.array([other_state['x'], other_state['y'], other_state['z']])
    m_other = other_state['mass']
    r = r_self - r_other
    dvdt = -m_other * r / np.linalg.norm(r)**3
    v_self = v_self + dvdt * time_step
    r_self = r_self + v_self * time_step

    return {
        "time": time + time_step,
        "timeStep": 5. + random() * 9.,
        "mass": self_state["mass"],
        "x": r_self[0],
        "y": r_self[1],
        "z": r_self[2],
        "vx": v_self[0],
        "vy": v_self[1],
        "vz": v_self[2],
    }
