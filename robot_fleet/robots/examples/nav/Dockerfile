FROM python:3.11-slim

WORKDIR /app

# Copy the entire project first
COPY ../../../.. /app/

# Install dependencies
RUN pip install -r requirements.txt

# Install the package in development mode
RUN pip install -e .

# Install development dependencies
RUN pip install pytest pytest-asyncio

# Default environment variables from pick_place.yaml
ENV ROBOT_ID=""
ENV PORT=5001
ENV LOG_LEVEL=info
ENV ARM_CONFIG=""
ENV GRIPPER_CONFIG=""
ENV PYTHONPATH=/app

# Expose the robot server port
EXPOSE 5001

# Run the robot server
CMD ["python", "-m", "robot_fleet.robots.examples.nav.nav_robot"] 