# dane-object-classification-worker

This is a worker that interacts with [DANE](https://github.com/CLARIAH/DANE) to receive its work.
Given a set of keyframes given by the [shot detection worker](https://github.com/CLARIAH/shot-detection-worker),
it applies a ResNet50 model to each keyframe, writing to a Elasticsearch database a `2048` dimensional embedding vector, and
a softmax vector over the `1000` ImageNet classes if `store_embeddings` is on. In any case it will store the top predictions and scores
(where top is determined based on the config specified `threshold`) as a DANE result.

# Installation

## Run locally:

Prerequisites:

This worker uses Pytorch as a DNN framework, to install it follow the instructions given here: https://pytorch.org/get-started/locally/

Other prerequisites:

- Python 3.10.x
- [Poetry](https://python-poetry.org/)

```sh
poetry install
poetry shell
python ./worker.py
```

## Docker

```sh
docker build -t dane-object-classification-worker .
```

# Configuration

Make sure the create, and fill in to match your environment, the following `config.yml`:

```yaml
RABBITMQ:
    HOST: your-rabbit-mq-host  # set to your rabbitMQ server
    PORT: 5672
    EXCHANGE: DANE-exchange
    RESPONSE_QUEUE: DANE-response-queue
    USER: guest # change this for production mode
    PASSWORD: guest # change this for production mode
ELASTICSEARCH:
    HOST: ["elasticsearch-host"]  # set to your elasticsearch host
    PORT: 9200
    USER: "" # change this for production mode
    PASSWORD: "" # change this for production mode
    SCHEME: http  # OR https
    INDEX: your-dane-index  # change to your liking
LOGGING:
    LEVEL: INFO
PATHS: # common settings for each DANE worker to define input/output dirs (with a common mount point)
    TEMP_FOLDER: "./mount" # directory is automatically created (use ./mount for local testing)
    OUT_FOLDER: "./mount" # directory is automatically created (use ./mount for local testing)
CLASSIFICATION:  # settings for this worker specifically
  STORE_EMBEDDINGS: false  # store the vector back to Elasticsearch
  BATCH_SIZE: 10  # batch size for the torch data loader
  LOAD_WORKERS: 2  # number of workers
  THRESHOLD: 0.6  # threshold of score
```


# Run

The best way to run is within a Kubernetes environment, together with the following:

- DANE-server
- dane-shot-detection-worker
- RabbitMQ server
- Elasticsearch cluster

Documentation on how to setup this environment will be linked later on
