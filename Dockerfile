FROM nvidia/cuda:11.3.1-base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 python3-pip python3-setuptools python3-dev

WORKDIR /app
COPY . /app

RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENTRYPOINT ["python3"]
CMD ["-u", "app.py"]