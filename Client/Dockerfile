FROM python:3.7

ADD client.py /
ADD motionsense.py /
ADD compression_utils.py /
add data /data

RUN pip install numpy pandas torch flwr torchvision sklearn

CMD ["python", "-u", "client.py"]