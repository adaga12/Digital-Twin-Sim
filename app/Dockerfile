# Running python
FROM python:3.12

# Create and set working directory
RUN mkdir /app
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the model file
COPY state_predictor_nn.keras /app/

# Copy the rest of the application code
COPY . .

# Verify the model file exists
RUN ls -l /app/state_predictor_nn.keras

# Run the simulation script to generate training data
RUN python3 /app/ml_model.py

# Set the command to run the Flask application
CMD [ "python3", "-m" , "flask", "run", "--debug", "--host=0.0.0.0", "--port=8000" ]