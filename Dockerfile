# Use a smaller base image
FROM python:3.10.11-alpine AS base

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install necessary dependencies for building packages
RUN apk add --no-cache build-base libffi-dev

# Install virtualenv
RUN pip install virtualenv

# Create a virtual environment
RUN virtualenv /venv

# Activate the virtual environment
ENV PATH="/venv/bin:$PATH"

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Use a lightweight runtime image
FROM python:3.10.11-alpine AS final

# Set the working directory in the container
WORKDIR /app

# Copy the virtual environment from the base image
COPY --from=base /venv /venv

# Activate the virtual environment
ENV PATH="/venv/bin:$PATH"

# Copy the remaining files to the container
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 5000

# Specify the command to run the container
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
