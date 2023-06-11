FROM python:3.10.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --default-timeout=100 -r requirements.txt

COPY . .

# CMD ["python", "app.py"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]