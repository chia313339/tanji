FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN apt update
RUN yes | apt install vim

COPY ./ /app
WORKDIR /app

ENV STATIC_URL /app/static
ENV STATIC_PATH /var/www/app/static

RUN pip install -r /app/requirements.txt