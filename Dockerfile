FROM docker.io/python:3.10

COPY ./ /src

# override this config in Kubernetes with a ConfigMap mounted as a volume to /root/.DANE
RUN mkdir /root/.DANE

# create the mountpoint for storing /input-files
RUN mkdir /mnt/dane-fs

WORKDIR /src

RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

CMD [ "python", "worker.py" ]
