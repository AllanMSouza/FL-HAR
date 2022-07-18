FROM python:3.7

ADD server.py /

RUN pip install numpy pandas torch flwr torchvision

CMD ["python", "-u", "server.py"]