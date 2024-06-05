FROM --platform=arm64 python:3.11.5

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

CMD ["python", "app.py"]