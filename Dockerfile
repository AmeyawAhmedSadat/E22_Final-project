FROM ubuntu:20.04
LABEL Maintainer="Ahmed"  email="st11078@gsom.spbu.ru"
RUN apt-get update -y
COPY . /opt/gsom_predictor
WORKDIR /opt/gsom_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app.py
