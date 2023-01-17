FROM python:3.8

WORKDIR /usr/app

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

WORKDIR /usr/app/src

CMD ["flask", "--debug", "run", "--host", "0.0.0.0"]
