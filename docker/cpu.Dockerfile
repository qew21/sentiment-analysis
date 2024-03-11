FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app

RUN pip install --no-cache-dir flask transformers -i https://mirrors.aliyun.com/pypi/simple

COPY . /app

WORKDIR /app

CMD ["python", "server.py"]