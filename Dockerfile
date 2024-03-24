FROM python:3.11.0

WORKDIR /app

COPY ./requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app
EXPOSE 5000
#CMD ["flask", "run", "--host", "0.0.0.0", "--port", "5000"]
ENTRYPOINT ["python", "/app/app.py"]



