FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for venv and build
RUN apt-get update && apt-get install -y python3-venv build-essential && rm -rf /var/lib/apt/lists/*

# Create venv in a clean location before copying files
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY ../../../.. /app/

# Set environment variables for robot config and id
ENV ROBOT_ID=moma
ENV CONFIG_FILE=/app/robot_fleet/robots/examples/moma/moma.yaml

# Install dependencies using venv's pip
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

ENV PYTHONPATH=/app

# Default environment variables from pick_place.yaml
ENV PORT=8002
ENV LOG_LEVEL=info
ENV ARM_CONFIG=""
ENV GRIPPER_CONFIG=""

# Expose the robot server port
EXPOSE 8002

# Run the robot server
CMD python robot_fleet/robots/examples/fake_robot_server.py --robot_id $ROBOT_ID --config $CONFIG_FILE 