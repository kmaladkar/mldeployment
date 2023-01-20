FROM python:3.8

RUN mkdir prediction-api 

WORKDIR /prediction-api

COPY ./service/requirements.txt /prediction-api/
RUN pip install -r requirements.txt

COPY ./models/model.pkl /prediction-api/
COPY ./service/app.py /prediction-api/
COPY ./templates/index.html /prediction-api/

CMD python /prediction-api/app.py