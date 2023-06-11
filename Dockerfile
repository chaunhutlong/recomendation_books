# Use the official Python image as the base image
FROM python:3.10.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

RUN pip install --upgrade pip

# Install the Python dependencies
RUN pip3 install --user --no-cache-dir -r requirements.txt

# Copy the remaining files to the container
COPY . .

# Expose the port on which the Flask app will run
EXPOSE 5000

# CMD ["python", "app.py"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]